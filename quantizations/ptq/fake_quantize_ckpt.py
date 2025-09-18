import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from collections.abc import MutableMapping
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


_RICH_SPEC = importlib.util.find_spec("rich")
if _RICH_SPEC:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    _RICH_CONSOLE = Console(highlight=False)
else:
    _RICH_CONSOLE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply uniform fake quantization to all weights in a checkpoint"
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
        "--min_bits",
        type=int,
        default=None,
        help="Lower bound for bit-width selection (inclusive)",
    )
    parser.add_argument(
        "--max_bits",
        type=int,
        default=None,
        help="Upper bound for bit-width selection (inclusive)",
    )
    parser.add_argument(
        "--bits_map",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file describing per-tensor bit-width overrides"
        ),
    )
    parser.add_argument(
        "--save_bits_map",
        type=str,
        default=None,
        help="Write the final per-tensor bit-width configuration to this JSON file",
    )
    parser.add_argument(
        "--interactive",
        "--tui",
        dest="interactive",
        action="store_true",
        help="Launch an interactive TUI to choose bit-widths before quantization",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the quantization summary without writing the checkpoint",
    )
    parser.set_defaults(interactive=False)
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
    return parser.parse_args()


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


_SKIP_TOKENS = {"skip", "none", "fp32", "float32", "no", "off"}


def validate_bit_bounds(
    min_bits: Optional[int], max_bits: Optional[int]
) -> None:
    if min_bits is not None and min_bits <= 0:
        raise ValueError("min_bits must be a positive integer")
    if max_bits is not None and max_bits <= 0:
        raise ValueError("max_bits must be a positive integer")
    if (
        min_bits is not None
        and max_bits is not None
        and min_bits > max_bits
    ):
        raise ValueError("min_bits cannot be greater than max_bits")


def normalize_bitwidth(
    value,
    min_bits: Optional[int] = None,
    max_bits: Optional[int] = None,
    *,
    allow_skip: bool = True,
) -> Tuple[Optional[int], bool]:
    """Normalize a bit-width value applying optional bounds."""

    if value is None:
        return None, False

    try:
        bits = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bit-width value: {value!r}") from exc

    if bits <= 0:
        if allow_skip:
            return None, True
        raise ValueError("Bit-width must be a positive integer")

    clamped = bits
    if min_bits is not None:
        clamped = max(clamped, min_bits)
    if max_bits is not None:
        clamped = min(clamped, max_bits)

    clamped = int(clamped)
    changed = clamped != bits

    if clamped <= 0:
        if allow_skip:
            return None, True
        raise ValueError("Bit-width must remain positive after clamping")

    return clamped, changed


def normalize_bitwidth_map(
    overrides: Dict[str, Optional[int]],
    min_bits: Optional[int],
    max_bits: Optional[int],
) -> Tuple[Dict[str, Optional[int]], Dict[str, Tuple[Optional[int], Optional[int]]]]:
    """Normalize overrides while tracking clamped or skipped values."""

    normalized: Dict[str, Optional[int]] = {}
    adjustments: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

    for name, value in overrides.items():
        try:
            normalized_value, changed = normalize_bitwidth(
                value, min_bits, max_bits, allow_skip=True
            )
        except ValueError as exc:
            raise ValueError(f"Invalid bit-width for tensor '{name}': {value!r}") from exc

        normalized[name] = normalized_value
        if changed:
            adjustments[name] = (value, normalized_value)

    return normalized, adjustments


def load_bitwidth_overrides(path: str) -> Dict[str, Optional[int]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Bit-width override file must contain a JSON object")

    overrides: Dict[str, Optional[int]] = {}

    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("Tensor names in the override map must be strings")

        raw_value = value
        if isinstance(raw_value, dict):
            if "bits" in raw_value:
                raw_value = raw_value["bits"]
            elif "bit_width" in raw_value:
                raw_value = raw_value["bit_width"]

        if isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered in _SKIP_TOKENS:
                overrides[key] = None
                continue

        if raw_value is None:
            overrides[key] = None
            continue

        try:
            overrides[key] = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid bit-width override for tensor '{key}': {raw_value!r}"
            ) from exc

    return overrides


def save_bitwidth_overrides(
    path: str, mapping: Dict[str, Optional[int]]
) -> None:
    serializable = {
        key: (int(value) if value is not None else None)
        for key, value in mapping.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(serializable.items())), handle, indent=2)


def iter_state_tensors(
    state_dict,
) -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(state_dict, torch.nn.Module):
        iterable = state_dict.state_dict().items()
    elif isinstance(state_dict, dict):
        iterable = state_dict.items()
    else:
        to_state_dict = getattr(state_dict, "state_dict", None)
        iterable = to_state_dict().items() if callable(to_state_dict) else ()

    for key, value in iterable:
        if torch.is_tensor(value):
            yield key, value


def estimate_checkpoint_sizes(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
    bitwidth_map: Dict[str, Optional[int]],
) -> Tuple[float, float]:
    """Estimate raw and quantized storage requirements for tensors."""

    original_bytes = 0.0
    quantized_bytes = 0.0

    for name, tensor in named_tensors:
        numel = tensor.numel()
        elem_bytes = tensor.element_size()
        original_bytes += numel * elem_bytes
        if torch.is_floating_point(tensor):
            bits = bitwidth_map.get(name)
            if bits is None:
                quantized_bytes += numel * elem_bytes
            else:
                quantized_bytes += numel * bits / 8.0
        else:
            quantized_bytes += numel * elem_bytes

    return original_bytes, quantized_bytes


def format_tensor_shape(tensor: torch.Tensor) -> str:
    if tensor.dim() == 0:
        return "()"
    return "×".join(str(dim) for dim in tensor.shape)


def apply_override(
    overrides: Dict[str, Optional[int]],
    name: str,
    value: Optional[int],
    use_default: bool,
    default_bits: int,
) -> None:
    if use_default:
        overrides.pop(name, None)
        return

    if value is None:
        overrides[name] = None
        return

    if value == default_bits:
        overrides.pop(name, None)
        return

    overrides[name] = value


def parse_override_value(
    token: str,
    min_bits: Optional[int],
    max_bits: Optional[int],
) -> Tuple[Optional[int], bool, List[str]]:
    """Parse a user-supplied override string."""

    text = token.strip()
    if not text:
        raise ValueError("Empty bit-width input")

    lowered = text.lower()
    if lowered == "default":
        return None, True, []

    if lowered in _SKIP_TOKENS:
        return None, False, []

    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid bit-width value: {token!r}") from exc

    normalized, changed = normalize_bitwidth(
        parsed, min_bits, max_bits, allow_skip=True
    )

    messages: List[str] = []
    if normalized is None:
        messages.append(
            "Bit-width <= 0 disables quantization for this tensor."
        )
    elif changed:
        messages.append(
            f"Bit-width clamped from {parsed} to {normalized} by the specified bounds."
        )

    return normalized, False, messages


def interactive_select_bitwidths(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
    default_bits: int,
    overrides: Dict[str, Optional[int]],
    min_bits: Optional[int],
    max_bits: Optional[int],
) -> Dict[str, Optional[int]]:
    float_entries = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_floating_point(tensor)
    ]

    if not float_entries:
        message = "No floating-point tensors were found for quantization."
        if _RICH_CONSOLE:
            _RICH_CONSOLE.print(f"[yellow]{message}[/yellow]")
        else:
            print(message)
        return overrides

    if _RICH_CONSOLE:
        return _interactive_select_bitwidths_rich(
            float_entries, default_bits, overrides, min_bits, max_bits
        )

    return _interactive_select_bitwidths_plain(
        float_entries, default_bits, overrides, min_bits, max_bits
    )


def _interactive_select_bitwidths_rich(
    entries: Sequence[Tuple[str, torch.Tensor]],
    default_bits: int,
    overrides: Dict[str, Optional[int]],
    min_bits: Optional[int],
    max_bits: Optional[int],
) -> Dict[str, Optional[int]]:
    overrides = dict(overrides)
    instructions = (
        "Commands: <index>=<bits>, <index>=skip, <index>=default, "
        "<index> (prompt), all=<bits>, all=skip, all=default, reset, done"
    )

    while True:
        table = Table(
            title="Per-tensor Bit-widths",
            title_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Tensor", style="magenta")
        table.add_column("Shape", style="green", no_wrap=True)
        table.add_column("DType", style="blue", no_wrap=True)
        table.add_column("Elements", style="white", justify="right")
        table.add_column("Selected", style="bright_white", justify="right")

        for idx, (name, tensor) in enumerate(entries):
            elements = f"{tensor.numel():,}"
            shape = format_tensor_shape(tensor)
            selected = overrides.get(name, default_bits)

            if selected is None:
                selected_display = "[bold red]skip[/bold red]"
            elif name in overrides:
                selected_display = f"[bold yellow]{selected}[/bold yellow]"
            else:
                selected_display = f"[green]{selected}[/green]"

            table.add_row(
                str(idx),
                name,
                shape,
                str(tensor.dtype),
                elements,
                selected_display,
            )

        _RICH_CONSOLE.print(table)
        _RICH_CONSOLE.print(f"[dim]{instructions}[/dim]")

        try:
            command = Prompt.ask("Selection", default="done")
        except (KeyboardInterrupt, EOFError):
            _RICH_CONSOLE.print(
                "\n[yellow]Stopping interactive selection (user request).[/yellow]"
            )
            break

        command = command.strip()
        if not command:
            continue

        lowered = command.lower()
        if lowered in {"done", "q", "quit", "exit"}:
            break
        if lowered == "help":
            _RICH_CONSOLE.print(instructions)
            continue
        if lowered == "reset":
            overrides.clear()
            continue

        if lowered.startswith("all="):
            value_str = command.split("=", 1)[1]
            try:
                value, use_default, messages = parse_override_value(
                    value_str, min_bits, max_bits
                )
            except ValueError as exc:
                _RICH_CONSOLE.print(f"[red]{exc}[/red]")
                continue

            for msg in messages:
                _RICH_CONSOLE.print(f"[yellow]{msg}[/yellow]")

            for name, _ in entries:
                apply_override(overrides, name, value, use_default, default_bits)

            continue

        if "=" in command:
            index_part, value_part = command.split("=", 1)
        else:
            index_part, value_part = command, None

        try:
            index = int(index_part.strip())
        except ValueError:
            _RICH_CONSOLE.print(
                f"[red]Invalid tensor index: {index_part!r}[/red]"
            )
            continue

        if not (0 <= index < len(entries)):
            _RICH_CONSOLE.print(
                f"[red]Tensor index {index} is out of range.[/red]"
            )
            continue

        name, tensor = entries[index]

        if value_part is None:
            current_value = overrides.get(name, default_bits)
            default_prompt = "skip" if current_value is None else str(current_value)
            try:
                value_part = Prompt.ask(
                    f"Bits for '{name}'", default=default_prompt
                )
            except (KeyboardInterrupt, EOFError):
                _RICH_CONSOLE.print(
                    "\n[yellow]No change made for that tensor.[/yellow]"
                )
                continue

        value_part = value_part.strip()
        if not value_part:
            continue

        try:
            value, use_default, messages = parse_override_value(
                value_part, min_bits, max_bits
            )
        except ValueError as exc:
            _RICH_CONSOLE.print(f"[red]{exc}[/red]")
            continue

        for msg in messages:
            _RICH_CONSOLE.print(f"[yellow]{msg}[/yellow]")

        apply_override(overrides, name, value, use_default, default_bits)

    return overrides


def _interactive_select_bitwidths_plain(
    entries: Sequence[Tuple[str, torch.Tensor]],
    default_bits: int,
    overrides: Dict[str, Optional[int]],
    min_bits: Optional[int],
    max_bits: Optional[int],
) -> Dict[str, Optional[int]]:
    overrides = dict(overrides)

    print(
        "Interactive bit-width selection. Enter an integer, 'skip' to keep "
        "a tensor in floating point, 'default' to remove an override, or "
        "press Enter to keep the current value. Type 'done' to finish early."
    )

    for idx, (name, tensor) in enumerate(entries):
        while True:
            current_value = overrides.get(name, default_bits)
            default_token = "skip" if current_value is None else str(current_value)
            prompt = (
                f"[{idx}] {name} (shape {format_tensor_shape(tensor)}, "
                f"dtype {tensor.dtype}) [{default_token}]: "
            )

            try:
                response = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                print("Stopping interactive selection (user request).")
                return overrides

            response = response.strip()
            if not response:
                break

            lowered = response.lower()
            if lowered in {"done", "q", "quit", "exit"}:
                return overrides

            try:
                value, use_default, messages = parse_override_value(
                    response, min_bits, max_bits
                )
            except ValueError as exc:
                print(f"  Invalid input: {exc}")
                continue

            for msg in messages:
                print(f"  {msg}")

            apply_override(overrides, name, value, use_default, default_bits)
            break

    return overrides


def compute_final_bitwidths(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
    default_bits: int,
    overrides: Dict[str, Optional[int]],
    min_bits: Optional[int],
    max_bits: Optional[int],
) -> Dict[str, Optional[int]]:
    final_map: Dict[str, Optional[int]] = {}

    for name, tensor in named_tensors:
        if not torch.is_floating_point(tensor):
            continue

        raw_value = overrides.get(name, default_bits)
        normalized, _ = normalize_bitwidth(
            raw_value, min_bits, max_bits, allow_skip=True
        )
        final_map[name] = normalized

    return final_map


def summarize_bitwidths(bitwidth_map: Dict[str, Optional[int]]) -> str:
    if not bitwidth_map:
        return "n/a"

    counts: Dict[int, int] = {}
    skipped = 0

    for value in bitwidth_map.values():
        if value is None:
            skipped += 1
        else:
            counts[value] = counts.get(value, 0) + 1

    if counts and skipped == 0 and len(counts) == 1:
        (bits,) = counts.keys()
        return f"{bits}-bit"

    segments: List[str] = [
        f"{bits}-bit x{count}" for bits, count in sorted(counts.items())
    ]

    if skipped:
        segments.append(f"skip x{skipped}")

    if not segments:
        return "skip all"

    return f"varies ({', '.join(segments)})"


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
    bitwidth_label: str,
    original_bytes: float,
    quantized_bytes: float,
) -> None:
    if _RICH_CONSOLE:
        if bitwidth_label:
            scheme_label = f"{scheme} ({bitwidth_label})"
        else:
            scheme_label = scheme

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

        panel = Panel.fit(
            table,
            title="[bold bright_white on blue] Fake PTQ [/bold bright_white on blue]",
            border_style="bright_blue",
            padding=(1, 2),
        )
        _RICH_CONSOLE.print(panel)
        return

    # Plain-text fallback when rich is unavailable.
    print("Quantization summary:")
    if bitwidth_label:
        print(f"  Scheme: {scheme}, bits: {bitwidth_label}")
    else:
        print(f"  Scheme: {scheme}")
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


def main():
    args = parse_args()

    def notify(message: str, style: Optional[str] = None) -> None:
        if _RICH_CONSOLE:
            if style:
                _RICH_CONSOLE.print(f"[{style}]{message}[/{style}]")
            else:
                _RICH_CONSOLE.print(message)
        else:
            print(message)

    try:
        validate_bit_bounds(args.min_bits, args.max_bits)
    except ValueError as exc:
        raise SystemExit(f"Invalid bit-width bounds: {exc}") from exc

    try:
        default_bits, default_changed = normalize_bitwidth(
            args.num_bits, args.min_bits, args.max_bits, allow_skip=False
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid --num_bits value: {exc}") from exc

    if default_changed:
        notify(
            f"--num_bits adjusted to {default_bits} to satisfy the provided bounds.",
            "yellow",
        )

    args.num_bits = default_bits

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

    named_tensors = list(iter_state_tensors(state_dict))
    float_tensor_names = {
        name for name, tensor in named_tensors if torch.is_floating_point(tensor)
    }

    overrides: Dict[str, Optional[int]] = {}
    if args.bits_map:
        try:
            loaded_overrides = load_bitwidth_overrides(args.bits_map)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Failed to load bit-width map: {exc}") from exc

        missing = sorted(set(loaded_overrides) - float_tensor_names)
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... (+{len(missing) - 5} more)"
            notify(
                "Ignoring overrides for tensors not present in the checkpoint: "
                f"{preview}",
                "yellow",
            )

        overrides = {
            name: loaded_overrides[name]
            for name in loaded_overrides
            if name in float_tensor_names
        }

        try:
            overrides, adjustments = normalize_bitwidth_map(
                overrides, args.min_bits, args.max_bits
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        if adjustments:
            items = sorted(adjustments.items())
            for name, (requested, normalized) in items[:5]:
                normalized_label = "skip" if normalized is None else str(normalized)
                notify(
                    f"Normalized override for '{name}': {requested!r} -> {normalized_label}",
                    "yellow",
                )
            if len(items) > 5:
                notify(
                    f"... and {len(items) - 5} more overrides were adjusted.",
                    "yellow",
                )

    if args.interactive:
        if not sys.stdin.isatty():
            notify(
                "Interactive selection requested, but no TTY is available. Skipping.",
                "yellow",
            )
        else:
            overrides = interactive_select_bitwidths(
                named_tensors, args.num_bits, overrides, args.min_bits, args.max_bits
            )

    overrides = {
        name: overrides[name]
        for name in overrides
        if name in float_tensor_names
    }

    final_bitwidths = compute_final_bitwidths(
        named_tensors, args.num_bits, overrides, args.min_bits, args.max_bits
    )

    original_bytes, quantized_bytes = estimate_checkpoint_sizes(
        named_tensors, final_bitwidths
    )

    bits_summary = summarize_bitwidths(final_bitwidths)

    for name, tensor in named_tensors:
        if not torch.is_floating_point(tensor):
            continue

        bits = final_bitwidths.get(name)
        if bits is None:
            continue

        state_dict[name] = fake_quant_tensor(tensor, bits, args.quantization)

    if args.save_bits_map:
        save_path = args.save_bits_map
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        try:
            save_bitwidth_overrides(save_path, final_bitwidths)
        except OSError as exc:
            raise SystemExit(f"Failed to save bit-width map: {exc}") from exc

        notify(
            f"Saved bit-width configuration to {os.path.abspath(save_path)}",
            "cyan",
        )

    print_quantization_summary(
        args.quantization,
        bits_summary,
        original_bytes,
        quantized_bytes,
    )

    out_dir = args.out_dir or f"{args.ckpt_dir}_ptq"

    if args.dry_run:
        notify(
            "Dry run: quantized checkpoint would be written to "
            f"{os.path.abspath(out_dir)}",
            "yellow",
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    meta_in = os.path.join(args.ckpt_dir, "meta.pkl")
    meta_out = os.path.join(out_dir, "meta.pkl")
    if os.path.exists(meta_in):
        shutil.copy(meta_in, meta_out)

    notify(
        f"Saved quantized checkpoint to {os.path.abspath(out_dir)}",
        "cyan",
    )

if __name__ == "__main__":
    main()
