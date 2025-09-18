import argparse
import importlib.util
import json
import math
import os
import shutil
import textwrap
from collections import Counter
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


_RICH_SPEC = importlib.util.find_spec("rich")
if _RICH_SPEC:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table

    _RICH_CONSOLE = Console(highlight=False)
else:
    _RICH_CONSOLE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply uniform fake quantization to all weights in a checkpoint. "
            "Supports optional per-tensor overrides and an interactive TUI."
        )
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and meta.pkl from a previous training run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write the quantized checkpoint (defaults to <ckpt_dir>_ptq)",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=8,
        help="Number of bits for uniform quantization",
    )
    parser.add_argument(
        "--per-tensor-bits",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Optional per-tensor bit-width overrides. Provide a path to a JSON file "
            "or an inline mapping such as 'tensor=4,other=8'."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Launch an interactive text UI to choose bit-widths for each tensor before "
            "quantization."
        ),
    )
    parser.add_argument(
        "--min-bits",
        type=int,
        default=1,
        help=(
            "Minimum allowed bit-width when selecting per-tensor values interactively. "
            "Use 0 to allow keeping tensors in floating point."
        ),
    )
    parser.add_argument(
        "--max-bits",
        type=int,
        default=16,
        help=(
            "Maximum allowed bit-width when selecting per-tensor values interactively."
        ),
    )
    parser.add_argument(
        "--tui-page-size",
        type=int,
        default=20,
        help="Number of tensors to display per page in the interactive TUI.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="symmetric",
        choices=("symmetric", "asymmetric"),
        help=(
            "Quantization scheme to use: symmetric signed (two's complement) or "
            "asymmetric unsigned"
        ),
    )
    args = parser.parse_args()
    if args.num_bits < 0:
        parser.error("--num_bits must be non-negative")
    if args.min_bits < 0:
        parser.error("--min-bits must be non-negative")
    if args.max_bits is not None and args.max_bits <= 0:
        parser.error("--max-bits must be positive")
    if args.max_bits is not None and args.min_bits > args.max_bits:
        parser.error("--min-bits cannot exceed --max-bits")
    if args.tui_page_size <= 0:
        parser.error("--tui-page-size must be positive")
    return args


@dataclass
class TensorConfigEntry:
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    default_bits: int
    bits: int


def _print_info(message: str) -> None:
    if _RICH_CONSOLE:
        _RICH_CONSOLE.print(f"[cyan]{message}[/cyan]")
    else:
        print(message)


def _print_warning(message: str) -> None:
    if _RICH_CONSOLE:
        _RICH_CONSOLE.print(f"[yellow]Warning:[/yellow] {message}")
    else:
        print(f"Warning: {message}")


def _format_bits_label(bits: int) -> str:
    if bits <= 0:
        return "fp32"
    return f"{bits}-bit"


def _parse_bits_value(value) -> int:
    if isinstance(value, bool):
        raise ValueError("Bit-width must be an integer, not a boolean value")
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"Bit-width must be an integer, got {value}")
            bits = int(round(value))
        else:
            bits = int(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Bit-width value cannot be empty")
        lowered = text.lower()
        if lowered in {"fp32", "float", "skip", "none"}:
            return 0
        bits = int(text, 0)
    if bits < 0:
        raise ValueError("Bit-width must be non-negative")
    return bits


def _parse_simple_mapping(text: str) -> Optional[Dict[str, int]]:
    mapping: Dict[str, int] = {}
    found_entry = False
    for raw_line in text.replace(",", "\n").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        found_entry = True
        if "=" in line:
            key, bits_str = line.split("=", 1)
        elif ":" in line:
            key, bits_str = line.split(":", 1)
        else:
            return None
        key = key.strip()
        bits_str = bits_str.strip()
        if not key:
            raise ValueError("Missing tensor name in per-tensor bit specification")
        if not bits_str:
            raise ValueError(f"Missing bit-width for tensor '{key}'")
        bits = _parse_bits_value(bits_str)
        mapping[key] = bits
    if not found_entry:
        return {}
    return mapping


def parse_per_tensor_bits(spec: Optional[str]) -> Dict[str, int]:
    if spec is None:
        return {}

    spec = spec.strip()
    if not spec:
        return {}

    data = None
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            mapping = _parse_simple_mapping(text)
            if mapping is None:
                raise ValueError(
                    "Unable to parse per-tensor bit specification file. "
                    "Use JSON or key=value pairs."
                )
            return mapping
    else:
        try:
            data = json.loads(spec)
        except json.JSONDecodeError:
            mapping = _parse_simple_mapping(spec)
            if mapping is None:
                raise ValueError(
                    "Unable to parse per-tensor bit specification. "
                    "Use JSON or comma-separated key=value pairs."
                )
            return mapping

    if not isinstance(data, dict):
        raise ValueError("Per-tensor bit specification must be a mapping")

    mapping: Dict[str, int] = {}
    for key, value in data.items():
        mapping[str(key)] = _parse_bits_value(value)
    return mapping


def iter_state_items(state_dict) -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(state_dict, torch.nn.Module):
        iterable = state_dict.state_dict().items()
    elif isinstance(state_dict, dict):
        iterable = state_dict.items()
    else:
        iterable = getattr(state_dict, "state_dict", lambda: {})().items()

    for key, value in iterable:
        if torch.is_tensor(value):
            yield key, value


def build_tensor_config_entries(
    state_dict, default_bits: int, overrides: Dict[str, int]
) -> List[TensorConfigEntry]:
    entries: List[TensorConfigEntry] = []
    for name, tensor in iter_state_items(state_dict):
        if not torch.is_floating_point(tensor):
            continue
        initial_bits = overrides.get(name, default_bits)
        dtype_str = str(tensor.dtype)
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str.split(".", 1)[1]
        entries.append(
            TensorConfigEntry(
                name=name,
                shape=tuple(tensor.shape),
                numel=tensor.numel(),
                dtype=dtype_str,
                default_bits=initial_bits,
                bits=initial_bits,
            )
        )
    return entries


def _format_shape(shape: Tuple[int, ...]) -> str:
    if not shape:
        return "scalar"
    return "×".join(str(dim) for dim in shape)


def _resolve_entry(
    entries: List[TensorConfigEntry], target: str
) -> Tuple[Optional[TensorConfigEntry], Optional[str]]:
    identifier = target.strip()
    if not identifier:
        return None, "Tensor identifier cannot be empty"

    if identifier.isdigit():
        index = int(identifier)
        if index < 1 or index > len(entries):
            return None, f"Index {index} is out of range (1-{len(entries)})"
        return entries[index - 1], None

    for entry in entries:
        if entry.name == identifier:
            return entry, None

    matches = [entry for entry in entries if identifier in entry.name]
    if not matches:
        return None, f"No tensor matching '{target}'"
    if len(matches) > 1:
        preview = ", ".join(entry.name for entry in matches[:5])
        if len(matches) > 5:
            preview += ", ..."
        return None, f"Ambiguous tensor name '{target}' (matches: {preview})"
    return matches[0], None


def _interactive_instruction_text(min_bits: int, max_bits: Optional[int]) -> str:
    if max_bits is None:
        if min_bits <= 0:
            allowed_line = (
                "Allowed bit-widths: 0 (keep float32) or any positive integer"
            )
        else:
            allowed_line = (
                f"Allowed bit-widths: {min_bits}+ (use 0 to keep float32)"
            )
    else:
        if min_bits <= 0:
            allowed_line = (
                f"Allowed bit-widths: 0 (keep float32) or 1-{max_bits}"
            )
        else:
            allowed_line = (
                f"Allowed bit-widths: {min_bits}-{max_bits} (use 0 to keep float32)"
            )

    return textwrap.dedent(
        f"""
        Commands:
          set <index|name> <bits>  Set bit-width for a tensor (use 0 to keep float32)
          all <bits>               Apply a bit-width to every tensor
          reset [<index|name>]     Reset all or a single tensor to the default value
          next / prev              Move between pages of tensors
          page <number>            Jump to a specific page (1-based)
          done / apply             Finish configuration and continue
          quit / cancel            Abort the operation
        {allowed_line}
        """
    ).strip()


def _render_tensor_table(
    entries: List[TensorConfigEntry],
    page: int,
    page_size: int,
    total_pages: int,
    instructions: str,
    status_message: str,
    status_style: str,
) -> None:
    total = len(entries)
    start = page * page_size
    end = min(start + page_size, total)

    if _RICH_CONSOLE:
        _RICH_CONSOLE.clear()
        instructions_panel = Panel(
            instructions,
            title="Per-tensor bit-width selection",
            border_style="bright_blue",
            padding=(0, 1),
            highlight=False,
        )
        _RICH_CONSOLE.print(instructions_panel)

        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Tensor", style="bright_white")
        table.add_column("Shape", style="green")
        table.add_column("DType", style="cyan")
        table.add_column("Elements", justify="right", style="yellow")
        table.add_column("Current", style="bright_white")
        table.add_column("Default", style="dim")

        for idx, entry in enumerate(entries[start:end], start=start + 1):
            shape_str = _format_shape(entry.shape)
            numel_str = f"{entry.numel:,}"
            bits_label = _format_bits_label(entry.bits)
            default_label = _format_bits_label(entry.default_bits)
            if entry.bits <= 0:
                bits_display = f"[dim]{bits_label}[/dim]"
            elif entry.bits != entry.default_bits:
                bits_display = f"[bold yellow]{bits_label}[/bold yellow]"
            else:
                bits_display = bits_label
            table.add_row(
                str(idx),
                entry.name,
                shape_str,
                entry.dtype,
                numel_str,
                bits_display,
                default_label,
            )

        table.caption = (
            f"Showing tensors {start + 1}-{end} of {total} "
            f"(page {page + 1}/{total_pages})"
        )
        table.caption_style = "dim"
        _RICH_CONSOLE.print(table)

        if status_message:
            _RICH_CONSOLE.print(f"[{status_style}]{status_message}[/{status_style}]")
        return

    print("\033c", end="")
    print("Per-tensor bit-width selection")
    print("=" * 72)
    print(instructions)
    print("")
    header = (
        f"{'#':>4}  {'Tensor':<48}  {'Shape':<24}  {'DType':<12}  "
        f"{'Elements':>12}  {'Bits':>8}  {'Default':>8}"
    )
    print(header)
    print("-" * len(header))
    for idx, entry in enumerate(entries[start:end], start=start + 1):
        name_display = entry.name if len(entry.name) <= 48 else entry.name[:45] + "..."
        shape_str = _format_shape(entry.shape)
        if len(shape_str) > 24:
            shape_str = shape_str[:21] + "..."
        bits_label = _format_bits_label(entry.bits)
        default_label = _format_bits_label(entry.default_bits)
        print(
            f"{idx:>4}  {name_display:<48}  {shape_str:<24}  {entry.dtype:<12}  "
            f"{entry.numel:>12,d}  {bits_label:>8}  {default_label:>8}"
        )
    print("-" * len(header))
    print(
        f"Showing tensors {start + 1}-{end} of {total} (page {page + 1}/{total_pages})"
    )
    if status_message:
        print(status_message)


def interactive_select_tensor_bits(
    entries: List[TensorConfigEntry],
    min_bits: int,
    max_bits: Optional[int],
    page_size: int,
) -> Dict[str, int]:
    if not entries:
        _print_warning(
            "No floating-point tensors were found for interactive configuration."
        )
        return {}

    page_size = max(page_size, 1)
    total_pages = max(1, math.ceil(len(entries) / page_size))
    page = 0
    instructions = _interactive_instruction_text(min_bits, max_bits)
    status_message = ""
    status_style = "cyan"

    def set_status(message: str, style: str = "cyan") -> None:
        nonlocal status_message, status_style
        status_message = message
        status_style = style

    _render_tensor_table(
        entries, page, page_size, total_pages, instructions, status_message, status_style
    )

    while True:
        try:
            raw = input("tui> ").strip()
        except EOFError:
            set_status("EOF received; applying current configuration.", "yellow")
            break
        except KeyboardInterrupt:
            raise SystemExit("Interactive configuration canceled by user.") from None

        if not raw:
            set_status("", "cyan")
            _render_tensor_table(
                entries,
                page,
                page_size,
                total_pages,
                instructions,
                status_message,
                status_style,
            )
            continue

        parts = raw.split()
        command = parts[0].lower()

        if command in {"done", "apply"}:
            set_status("Applying selected configuration.", "green")
            break

        if command in {"quit", "exit", "cancel"}:
            raise SystemExit("Interactive configuration canceled by user.")

        if command in {"next", "n"}:
            if page + 1 < total_pages:
                page += 1
                set_status(f"Moved to page {page + 1}/{total_pages}.")
            else:
                set_status("Already on the last page.", "yellow")
        elif command in {"prev", "p"}:
            if page > 0:
                page -= 1
                set_status(f"Moved to page {page + 1}/{total_pages}.")
            else:
                set_status("Already on the first page.", "yellow")
        elif command == "page":
            if len(parts) < 2:
                set_status("Usage: page <number>", "red")
            else:
                try:
                    new_page = int(parts[1]) - 1
                except ValueError:
                    set_status("Page number must be an integer.", "red")
                else:
                    if 0 <= new_page < total_pages:
                        page = new_page
                        set_status(f"Moved to page {page + 1}/{total_pages}.")
                    else:
                        set_status("Page number out of range.", "red")
        elif command == "set":
            if len(parts) < 3:
                set_status("Usage: set <index|name> <bits>", "red")
            else:
                target = parts[1]
                bits_str = parts[2]
                try:
                    bits = _parse_bits_value(bits_str)
                except ValueError as exc:
                    set_status(str(exc), "red")
                else:
                    if bits > 0 and bits < min_bits:
                        set_status(
                            f"Bit-width must be at least {min_bits} for quantized tensors.",
                            "red",
                        )
                    elif max_bits is not None and bits > 0 and bits > max_bits:
                        set_status(
                            f"Bit-width must be at most {max_bits} for quantized tensors.",
                            "red",
                        )
                    else:
                        entry, error = _resolve_entry(entries, target)
                        if entry is None:
                            set_status(error or "Unknown tensor", "red")
                        else:
                            entry.bits = bits
                            set_status(
                                f"Set {entry.name} to {_format_bits_label(bits)}.",
                                "green",
                            )
        elif command == "all":
            if len(parts) < 2:
                set_status("Usage: all <bits>", "red")
            else:
                try:
                    bits = _parse_bits_value(parts[1])
                except ValueError as exc:
                    set_status(str(exc), "red")
                else:
                    if bits > 0 and bits < min_bits:
                        set_status(
                            f"Bit-width must be at least {min_bits} for quantized tensors.",
                            "red",
                        )
                    elif max_bits is not None and bits > 0 and bits > max_bits:
                        set_status(
                            f"Bit-width must be at most {max_bits} for quantized tensors.",
                            "red",
                        )
                    else:
                        for entry in entries:
                            entry.bits = bits
                        set_status(
                            f"Applied {_format_bits_label(bits)} to {len(entries)} tensor(s).",
                            "green",
                        )
        elif command == "reset":
            if len(parts) == 1:
                for entry in entries:
                    entry.bits = entry.default_bits
                set_status("Reset all tensors to their default bit-widths.", "green")
            else:
                target = parts[1]
                entry, error = _resolve_entry(entries, target)
                if entry is None:
                    set_status(error or "Unknown tensor", "red")
                else:
                    entry.bits = entry.default_bits
                    set_status(
                        f"Reset {entry.name} to {_format_bits_label(entry.default_bits)}.",
                        "green",
                    )
        elif command in {"help", "?"}:
            set_status("Help is shown above.")
        else:
            set_status(f"Unknown command: {command}", "red")

        _render_tensor_table(
            entries,
            page,
            page_size,
            total_pages,
            instructions,
            status_message,
            status_style,
        )

    if _RICH_CONSOLE:
        _RICH_CONSOLE.print("[green]Interactive configuration complete.[/green]")
    else:
        print("Interactive configuration complete.")

    return {entry.name: entry.bits for entry in entries}


def _fake_quant_symmetric(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    # Signed two's-complement style quantization covering
    #   qmin = -2^{B-1} and qmax = 2^{B-1} - 1
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -1 << (num_bits - 1)
    if qmax <= 0:
        return tensor

    if tensor.numel() == 0:
        return tensor

    max_abs = tensor.abs().max()
    if max_abs.numel() == 0:
        return tensor
    max_abs_val = max_abs.item()
    if max_abs_val == 0.0 or not math.isfinite(max_abs_val):
        return tensor

    scale = max_abs_val / qmax
    if scale == 0.0 or not math.isfinite(scale):
        return tensor

    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return (q * scale).to(tensor.dtype)


def _fake_quant_asymmetric(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    # Unsigned quantization with range [0, 2^{B}-1]
    qmin = 0
    qmax = (1 << num_bits) - 1
    if qmax <= qmin:
        return tensor

    if tensor.numel() == 0:
        return tensor

    # min/max provide scalar tensors; handle degenerate ranges gracefully
    min_val = tensor.min()
    max_val = tensor.max()
    if min_val.numel() == 0 or max_val.numel() == 0:
        return tensor

    min_float = min_val.item()
    max_float = max_val.item()
    if not (math.isfinite(min_float) and math.isfinite(max_float)):
        return tensor
    if max_float <= min_float:
        return tensor

    scale = (max_float - min_float) / float(qmax - qmin)
    if scale == 0.0 or not math.isfinite(scale):
        return tensor

    zero_point = qmin - round(min_float / scale)
    zero_point = max(qmin, min(qmax, int(zero_point)))

    q = torch.round(tensor / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return ((q - zero_point) * scale).to(tensor.dtype)


def fake_quant_tensor(
    tensor: torch.Tensor, num_bits: int, scheme: str
) -> torch.Tensor:
    """Uniform quantize then dequantize a tensor."""

    if not torch.is_floating_point(tensor):
        return tensor

    if num_bits <= 0:
        return tensor

    if scheme == "symmetric":
        return _fake_quant_symmetric(tensor, num_bits)
    if scheme == "asymmetric":
        return _fake_quant_asymmetric(tensor, num_bits)

    raise ValueError(f"Unsupported quantization scheme: {scheme}")


def iter_state_tensors(state_dict) -> Iterable[torch.Tensor]:
    for _, tensor in iter_state_items(state_dict):
        yield tensor


def estimate_checkpoint_sizes(
    state_dict,
    default_num_bits: int,
    tensor_bitwidths: Optional[Dict[str, int]] = None,
) -> Tuple[float, float]:
    """Estimate raw and quantized storage requirements for tensors in a state dict."""

    tensor_bitwidths = tensor_bitwidths or {}

    original_bytes = 0.0
    quantized_bytes = 0.0

    for name, tensor in iter_state_items(state_dict):
        numel = tensor.numel()
        elem_bytes = tensor.element_size()
        original = numel * elem_bytes
        original_bytes += original

        if torch.is_floating_point(tensor):
            bits = tensor_bitwidths.get(name, default_num_bits)
            if bits is None or bits <= 0:
                quantized_bytes += original
            else:
                quantized_bytes += numel * int(bits) / 8.0
        else:
            quantized_bytes += original

    return original_bytes, quantized_bytes


def _size_breakdown(num_bytes: float) -> Tuple[str, str, str, str]:
    kb = num_bytes / 1024.0
    mb = kb / 1024.0
    gb = mb / 1024.0
    return (
        f"{num_bytes:,.0f} bytes",
        f"{kb:,.2f} KB",
        f"{mb:,.2f} MB",
        f"{gb:,.4f} GB",
    )


def format_size(num_bytes: float) -> str:
    base, kb, mb, gb = _size_breakdown(num_bytes)
    return f"{base} ({kb} / {mb} / {gb})"


def _format_size_rich(
    num_bytes: float,
    value_style: str = "white",
    detail_style: str = "dim",
) -> str:
    base, kb, mb, gb = _size_breakdown(num_bytes)
    return (
        f"[{value_style}]{base}[/{value_style}]\n"
        f"[{detail_style}]{kb} | {mb} | {gb}[/{detail_style}]"
    )


def print_quantization_summary(
    scheme: str,
    num_bits: int,
    original_bytes: float,
    quantized_bytes: float,
    tensor_bitwidths: Optional[Dict[str, int]] = None,
) -> None:
    bit_counts: Counter[int] = Counter()
    skipped_count = 0
    if tensor_bitwidths:
        for bits in tensor_bitwidths.values():
            if bits is None or bits <= 0:
                skipped_count += 1
            else:
                bit_counts[int(bits)] += 1

    if _RICH_CONSOLE:
        scheme_label = f"{scheme} ({num_bits}-bit)"
        table = Table(
            title="Quantization Summary",
            title_style="bold magenta",
            show_header=False,
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        table.add_column("Metric", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="bright_white")
        table.add_row("Scheme", f"[bold green]{scheme_label}[/bold green]")
        table.add_row(
            "Original Size",
            _format_size_rich(original_bytes, value_style="bright_white"),
        )
        table.add_row(
            "Quantized Size",
            _format_size_rich(quantized_bytes, value_style="cyan"),
        )

        if original_bytes > 0:
            if quantized_bytes > 0:
                ratio = original_bytes / quantized_bytes
                pct_of_original = (quantized_bytes / original_bytes) * 100.0
                reduction_pct = 100.0 - pct_of_original
                bytes_saved = max(original_bytes - quantized_bytes, 0.0)
                table.add_row("Compression", f"[bold green]{ratio:.2f}x[/bold green]")
                table.add_row(
                    "Size Reduction",
                    f"[green]{reduction_pct:.2f}%[/green]",
                )
                table.add_row(
                    "Remaining Size",
                    f"[yellow]{pct_of_original:.2f}%[/yellow] of original",
                )
                table.add_row(
                    "Bytes Saved",
                    _format_size_rich(bytes_saved, value_style="bright_green"),
                )
            else:
                table.add_row(
                    "Compression",
                    "[bold green]∞[/bold green] ([green]100.00% size reduction[/green])",
                )
                table.add_row(
                    "Remaining Size",
                    "[yellow]0.00%[/yellow] of original",
                )
                bytes_saved = max(original_bytes - quantized_bytes, 0.0)
                table.add_row(
                    "Bytes Saved",
                    _format_size_rich(bytes_saved, value_style="bright_green"),
                )
        else:
            table.add_row("Compression", "[yellow]n/a[/yellow]")

        renderables = [table]
        if bit_counts or skipped_count:
            bit_table = Table(
                title="Bit-width Usage",
                title_style="bold magenta",
                header_style="bold cyan",
                box=box.SIMPLE_HEAVY,
                expand=True,
            )
            bit_table.add_column("Bit-width", justify="center", style="bright_white")
            bit_table.add_column("Tensors", justify="right", style="cyan")
            for bits, count in sorted(bit_counts.items()):
                bit_table.add_row(f"{bits}-bit", str(count))
            if skipped_count:
                bit_table.add_row("fp32", str(skipped_count))
            renderables.append(bit_table)

        panel_content = renderables[0] if len(renderables) == 1 else Group(*renderables)

        panel = Panel.fit(
            panel_content,
            title="[bold bright_white on blue] Fake PTQ [/bold bright_white on blue]",
            border_style="bright_blue",
            padding=(1, 2),
        )
        _RICH_CONSOLE.print(panel)
        return

    # Plain-text fallback when rich is unavailable.
    print("Quantization summary:")
    print(f"  Scheme: {scheme}, bits: {num_bits}")
    print("  Estimated checkpoint size before quantization:")
    print(f"    {format_size(original_bytes)}")
    print("  Estimated checkpoint size after quantization:")
    print(f"    {format_size(quantized_bytes)}")
    if original_bytes > 0:
        if quantized_bytes > 0:
            ratio = original_bytes / quantized_bytes
            pct_of_original = (quantized_bytes / original_bytes) * 100.0
            reduction_pct = 100.0 - pct_of_original
            bytes_saved = max(original_bytes - quantized_bytes, 0.0)
            print(
                "  Estimated compression factor:",
                f" {ratio:.2f}x ({reduction_pct:.2f}% size reduction,",
                f"{pct_of_original:.2f}% of original size)",
            )
            print(f"  Bytes saved: {format_size(bytes_saved)}")
        else:
            print("  Estimated compression factor: ∞ (100.00% size reduction)")
            print("  Remaining size: 0.00% of original")
            bytes_saved = max(original_bytes - quantized_bytes, 0.0)
            print(f"  Bytes saved: {format_size(bytes_saved)}")
    else:
        print("  Estimated compression factor: n/a")

    if bit_counts or skipped_count:
        print("  Per-tensor bit-widths:")
        for bits, count in sorted(bit_counts.items()):
            print(f"    {bits}-bit: {count} tensor(s)")
        if skipped_count:
            print(f"    fp32 (skipped): {skipped_count} tensor(s)")

def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_obj = checkpoint["model"]
    else:
        state_obj = checkpoint

    if isinstance(state_obj, MutableMapping):
        state_dict = state_obj
    else:
        to_state_dict = getattr(state_obj, "state_dict", None)
        if callable(to_state_dict):
            state_dict = to_state_dict()
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint["model"] = state_dict
            else:
                checkpoint = state_dict
        else:
            raise TypeError(
                "Unsupported checkpoint format: expected a mapping for the model state"
            )

    try:
        overrides = parse_per_tensor_bits(args.per_tensor_bits)
    except ValueError as exc:
        raise SystemExit(f"Failed to parse --per-tensor-bits: {exc}") from None
    except OSError as exc:
        raise SystemExit(f"Unable to read --per-tensor-bits: {exc}") from None

    valid_overrides: Dict[str, int] = {}
    for name, bits in overrides.items():
        if name not in state_dict:
            _print_warning(f"Ignoring override for unknown tensor '{name}'")
            continue
        value = state_dict[name]
        if not torch.is_tensor(value):
            _print_warning(f"Ignoring override for non-tensor entry '{name}'")
            continue
        if not torch.is_floating_point(value):
            _print_warning(
                f"Ignoring override for non-floating tensor '{name}' (dtype: {value.dtype})"
            )
            continue
        valid_overrides[name] = bits

    if valid_overrides:
        _print_info(
            f"Loaded per-tensor overrides for {len(valid_overrides)} tensor(s)."
        )

    entries = build_tensor_config_entries(state_dict, args.num_bits, valid_overrides)

    if entries:
        _print_info(
            f"Detected {len(entries)} floating-point tensor(s) available for quantization."
        )

    positive_bits = [entry.bits for entry in entries if entry.bits > 0]
    if args.num_bits > 0:
        positive_bits.append(args.num_bits)

    min_positive = min(positive_bits) if positive_bits else None
    max_positive = max(positive_bits) if positive_bits else None

    effective_min_bits = args.min_bits if args.min_bits > 0 else (min_positive or 1)
    if min_positive is not None and min_positive < effective_min_bits:
        if args.min_bits > 0:
            _print_info(
                f"Lowering minimum interactive bit-width to {min_positive} to accommodate overrides."
            )
        effective_min_bits = min_positive

    effective_max_bits = args.max_bits
    if max_positive is not None:
        if effective_max_bits is None:
            effective_max_bits = max_positive
        elif max_positive > effective_max_bits:
            _print_info(
                f"Raising maximum interactive bit-width to {max_positive} to accommodate overrides."
            )
            effective_max_bits = max_positive

    if effective_max_bits is not None and effective_max_bits < effective_min_bits:
        effective_max_bits = effective_min_bits

    if args.interactive:
        tensor_bitwidths = interactive_select_tensor_bits(
            entries, effective_min_bits, effective_max_bits, args.tui_page_size
        )
        if not tensor_bitwidths and entries:
            tensor_bitwidths = {entry.name: entry.bits for entry in entries}
    else:
        tensor_bitwidths = {entry.name: entry.bits for entry in entries}

    original_bytes, quantized_bytes = estimate_checkpoint_sizes(
        state_dict, args.num_bits, tensor_bitwidths
    )

    applied_tensor_bits: Dict[str, int] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if not torch.is_floating_point(value):
            continue
        bits = tensor_bitwidths.get(key, args.num_bits)
        applied_tensor_bits[key] = bits
        if bits is None or bits <= 0:
            continue
        state_dict[key] = fake_quant_tensor(value, int(bits), args.quantization)

    if applied_tensor_bits:
        quantized_count = sum(
            1 for bits in applied_tensor_bits.values() if bits and bits > 0
        )
        skipped_count = sum(
            1 for bits in applied_tensor_bits.values() if bits is None or bits <= 0
        )
        _print_info(
            f"Configured per-tensor bit-widths for {len(applied_tensor_bits)} tensor(s): "
            f"{quantized_count} quantized, {skipped_count} kept as fp32."
        )

    out_dir = args.out_dir or f"{args.ckpt_dir}_ptq"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    meta_in = os.path.join(args.ckpt_dir, "meta.pkl")
    meta_out = os.path.join(out_dir, "meta.pkl")
    if os.path.exists(meta_in):
        shutil.copy(meta_in, meta_out)

    print_quantization_summary(
        args.quantization,
        args.num_bits,
        original_bytes,
        quantized_bytes,
        applied_tensor_bits if applied_tensor_bits else None,
    )

    if _RICH_CONSOLE:
        _RICH_CONSOLE.print(
            f"[cyan]Saved quantized checkpoint to[/cyan] "
            f"[bold]{os.path.abspath(out_dir)}[/bold]"
        )
    else:
        print(f"Saved quantized checkpoint to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
