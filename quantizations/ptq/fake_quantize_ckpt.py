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


def _describe_allowed_bits(min_bits: int, max_bits: Optional[int]) -> str:
    if max_bits is None:
        if min_bits <= 0:
            return "Allowed bit-widths: 0 (keep float32) or any positive integer"
        return f"Allowed bit-widths: {min_bits}+ (use 0 to keep float32)"

    if min_bits <= 0:
        return f"Allowed bit-widths: 0 (keep float32) or 1-{max_bits}"

    return f"Allowed bit-widths: {min_bits}-{max_bits} (use 0 to keep float32)"


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

    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container, Horizontal, Vertical
        from textual.screen import ModalScreen
        from textual.widgets import Button, DataTable, Footer, Header, Input, Static
    except ImportError as exc:
        raise SystemExit(
            "Interactive mode requires the 'textual' package from Textualize. "
            "Install it with `pip install textual`."
        ) from exc

    from rich.text import Text

    allowed_line = _describe_allowed_bits(min_bits, max_bits)
    instructions_text = textwrap.dedent(
        f"""
        [b]Controls[/b]
        • Arrow keys / PageUp / PageDown: Navigate tensors
        • [b]E[/b]: Edit the selected tensor
        • [b]A[/b]: Apply a bit-width to every tensor
        • [b]R[/b]: Reset the selected tensor to its default
        • [b]Shift+R[/b]: Reset all tensors to their defaults
        • [b]Enter[/b]: Apply the current configuration and exit
        • [b]Q[/b] or [b]Esc[/b]: Cancel without applying changes
        [italic]{allowed_line}[/italic]
        """
    ).strip()

    cancel_sentinel = object()

    class BitPrompt(ModalScreen[Optional[str]]):
        DEFAULT_CSS = """
        BitPrompt {
            align: center middle;
        }

        #dialog {
            width: 60%;
            max-width: 80;
            padding: 2 3;
            background: $surface;
            border: round $primary;
        }

        #prompt-text {
            padding-bottom: 1;
        }

        #prompt-actions {
            padding-top: 1;
            gap: 1;
        }
        """

        def __init__(self, title: str, message: str, initial: str = "") -> None:
            super().__init__()
            self._title = title
            self._message = message
            self._initial = initial

        def compose(self) -> ComposeResult:
            yield Container(
                Vertical(
                    Static(f"[b]{self._title}[/b]", id="prompt-title"),
                    Static(self._message, id="prompt-text"),
                    Input(value=self._initial, placeholder="Enter bit-width"),
                    Horizontal(
                        Button("Apply", id="apply", variant="success"),
                        Button("Cancel", id="cancel", variant="default"),
                        id="prompt-actions",
                    ),
                ),
                id="dialog",
            )

        async def on_mount(self) -> None:
            input_widget = self.query_one(Input)
            await input_widget.focus()
            input_widget.action_select_all()

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.dismiss(None)
            else:
                value = self.query_one(Input).value.strip()
                self.dismiss(value)

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            self.dismiss(event.value.strip())

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.dismiss(None)

    class TensorBitsApp(App[Dict[str, int]]):
        CSS = """
        Screen {
            layout: vertical;
        }

        #content {
            layout: vertical;
            height: 1fr;
            padding: 0 1;
        }

        #instructions {
            padding: 1 2;
            border: round $primary;
        }

        #main-area {
            layout: horizontal;
            height: 1fr;
            gap: 1;
        }

        #table-container {
            width: 3fr;
        }

        #side-panel {
            width: 2fr;
            min-width: 32;
            padding: 1 2;
            border: round $surface;
        }

        #status {
            min-height: 3;
            padding: 1 2;
            border: round $primary;
        }

        #summary {
            padding-top: 1;
        }
        """

        BINDINGS = [
            Binding("e", "edit_selected", "Edit Selected"),
            Binding("a", "edit_all", "Set All"),
            Binding("r", "reset_selected", "Reset Selected"),
            Binding("shift+r", "reset_all", "Reset All"),
            Binding("enter", "apply", "Apply"),
            Binding("ctrl+s", "apply", "Apply"),
            Binding("q", "cancel", "Cancel"),
            Binding("escape", "cancel", "Cancel"),
        ]

        def __init__(
            self,
            tensor_entries: List[TensorConfigEntry],
            instructions: str,
            allowed: str,
            minimum: int,
            maximum: Optional[int],
            page_size_hint: int,
        ) -> None:
            super().__init__()
            self.entries = tensor_entries
            self.instructions_text = instructions
            self.allowed_line = allowed
            self.min_bits = minimum
            self.max_bits = maximum
            self.page_size = max(page_size_hint, 1)
            self.entry_map = {entry.name: entry for entry in tensor_entries}

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield Container(
                Static(self.instructions_text, id="instructions"),
                Horizontal(
                    Container(DataTable(id="tensor-table"), id="table-container"),
                    Container(
                        Static("", id="details"),
                        Static("", id="summary"),
                        id="side-panel",
                    ),
                    id="main-area",
                ),
                Static("", id="status"),
                id="content",
            )
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one(DataTable)
            table.add_column("#", key="index", width=6, header_style="bold cyan")
            table.add_column("Tensor", key="tensor", header_style="bold magenta", width=40)
            table.add_column("Shape", key="shape", width=20)
            table.add_column("DType", key="dtype", width=10)
            table.add_column("Elements", key="elements", width=12)
            table.add_column("Current", key="current", width=12)
            table.add_column("Default", key="default", width=12)
            table.cursor_type = "row"
            table.zebra_stripes = True
            table.styles.height = max(self.page_size + 6, 12)

            for idx, entry in enumerate(self.entries, start=1):
                table.add_row(
                    str(idx),
                    entry.name,
                    _format_shape(entry.shape),
                    entry.dtype,
                    f"{entry.numel:,}",
                    self._bits_text(entry),
                    _format_bits_label(entry.default_bits),
                    key=entry.name,
                )

            if self.entries:
                table.focus()
                table.move_cursor(row=0, column=0)

            self._update_summary()
            self._update_details(self._get_selected_entry())
            self._set_status("Select a tensor and press E to edit it.", "cyan")

        def _bits_text(self, entry: TensorConfigEntry) -> Text:
            label = _format_bits_label(entry.bits)
            text = Text(label)
            if entry.bits <= 0:
                text.stylize("dim")
            elif entry.bits != entry.default_bits:
                text.stylize("bold yellow")
            return text

        def _set_status(self, message: str, style: str = "cyan") -> None:
            status = self.query_one("#status", Static)
            if message:
                status.update(Text(message, style=style))
            else:
                status.update(Text(" "))

        def _update_summary(self) -> None:
            summary = self.query_one("#summary", Static)
            total = len(self.entries)
            counts = Counter()
            skipped = 0
            for entry in self.entries:
                if entry.bits <= 0:
                    skipped += 1
                else:
                    counts[int(entry.bits)] += 1

            text = Text()
            text.append(f"Total tensors: {total}\n", style="bold white")
            if counts:
                text.append("Quantized:\n", style="bold magenta")
                for bits, count in sorted(counts.items()):
                    text.append(
                        f"  • {_format_bits_label(bits)}: {count}\n", style="cyan"
                    )
            if skipped:
                text.append(f"fp32 (kept): {skipped}\n", style="yellow")
            summary.update(text if text.plain else Text("No tensors", style="dim"))

        def _update_details(self, entry: Optional[TensorConfigEntry]) -> None:
            panel = self.query_one("#details", Static)
            if entry is None:
                panel.update(Text("No tensor selected.", style="dim"))
                return

            text = Text()
            text.append(f"{entry.name}\n", style="bold magenta")
            text.append(f"Shape: {_format_shape(entry.shape)}\n", style="cyan")
            text.append(f"DType: {entry.dtype}\n")
            text.append(f"Elements: {entry.numel:,}\n")
            text.append(
                f"Current: {_format_bits_label(entry.bits)}\n", style="green"
            )
            text.append(
                f"Default: {_format_bits_label(entry.default_bits)}",
                style="dim",
            )
            panel.update(text)

        def _get_selected_entry(self) -> Optional[TensorConfigEntry]:
            table = self.query_one(DataTable)
            row_key = getattr(table, "cursor_row_key", None)
            if row_key is None:
                cursor_row = getattr(table, "cursor_row", None)
                if cursor_row is None:
                    return None
                row_key = table.row_key(cursor_row)
            return self.entry_map.get(row_key)

        async def _prompt_for_bits(
            self, title: str, message: str, initial: str
        ) -> Optional[int]:
            current = initial
            while True:
                result = await self.push_screen(
                    BitPrompt(title, f"{message}\n{self.allowed_line}", current)
                )
                if result is None:
                    return None
                try:
                    bits = _parse_bits_value(result)
                except ValueError as exc:
                    self._set_status(str(exc), "red")
                    current = result
                    continue
                error = self._validate_bits(bits)
                if error:
                    self._set_status(error, "red")
                    current = result
                    continue
                return int(bits)

        def _validate_bits(self, bits: int) -> Optional[str]:
            if bits > 0 and bits < self.min_bits:
                return f"Bit-width must be at least {self.min_bits} for quantized tensors."
            if self.max_bits is not None and bits > 0 and bits > self.max_bits:
                return f"Bit-width must be at most {self.max_bits} for quantized tensors."
            return None

        def _refresh_entry(self, entry: TensorConfigEntry) -> None:
            table = self.query_one(DataTable)
            table.update_cell(entry.name, "current", self._bits_text(entry))

        def _refresh_table(self) -> None:
            for entry in self.entries:
                self._refresh_entry(entry)
            self._update_summary()
            self._update_details(self._get_selected_entry())

        async def action_edit_selected(self) -> None:
            entry = self._get_selected_entry()
            if entry is None:
                self._set_status("Select a tensor first.", "yellow")
                return
            initial = "0" if entry.bits <= 0 else str(entry.bits)
            bits = await self._prompt_for_bits(
                "Set bit-width",
                f"Enter a bit-width for '{entry.name}'.",
                initial,
            )
            if bits is None:
                self._set_status("Edit canceled.", "yellow")
                return
            entry.bits = bits
            self._refresh_entry(entry)
            self._update_summary()
            self._update_details(entry)
            self._set_status(
                f"Set {entry.name} to {_format_bits_label(bits)}.",
                "green",
            )

        async def action_edit_all(self) -> None:
            if not self.entries:
                self._set_status("No tensors to update.", "yellow")
                return
            initial = "0" if self.entries[0].bits <= 0 else str(self.entries[0].bits)
            bits = await self._prompt_for_bits(
                "Set bit-width for all tensors",
                "Enter a bit-width to apply to every tensor.",
                initial,
            )
            if bits is None:
                self._set_status("Bulk update canceled.", "yellow")
                return
            for entry in self.entries:
                entry.bits = bits
            self._refresh_table()
            self._set_status(
                f"Applied {_format_bits_label(bits)} to {len(self.entries)} tensor(s).",
                "green",
            )

        def action_reset_selected(self) -> None:
            entry = self._get_selected_entry()
            if entry is None:
                self._set_status("Select a tensor first.", "yellow")
                return
            entry.bits = entry.default_bits
            self._refresh_entry(entry)
            self._update_summary()
            self._update_details(entry)
            self._set_status(
                f"Reset {entry.name} to {_format_bits_label(entry.default_bits)}.",
                "green",
            )

        def action_reset_all(self) -> None:
            for entry in self.entries:
                entry.bits = entry.default_bits
            self._refresh_table()
            self._set_status("Reset all tensors to their defaults.", "green")

        def action_apply(self) -> None:
            result = {entry.name: entry.bits for entry in self.entries}
            self._set_status("Applying configuration...", "green")
            self.exit(result)

        def action_cancel(self) -> None:
            self.exit(cancel_sentinel)

        def on_data_table_row_highlighted(
            self, event: DataTable.RowHighlighted
        ) -> None:
            row_key = getattr(event, "row_key", None)
            if row_key is None:
                return
            entry = self.entry_map.get(row_key)
            self._update_details(entry)

    app = TensorBitsApp(entries, instructions_text, allowed_line, min_bits, max_bits, page_size)
    result = app.run()
    if result is cancel_sentinel:
        raise SystemExit("Interactive configuration canceled by user.")
    if not isinstance(result, dict):
        return {entry.name: entry.bits for entry in entries}
    return result


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
