import argparse
import importlib.util
import math
import os
import shutil
from collections.abc import MutableMapping
from typing import Iterable, Tuple

import torch


_RICH_SPEC = importlib.util.find_spec("rich")
if _RICH_SPEC:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
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


def iter_state_tensors(state_dict) -> Iterable[torch.Tensor]:
    if isinstance(state_dict, torch.nn.Module):
        iterable = state_dict.state_dict().values()
    elif isinstance(state_dict, dict):
        iterable = state_dict.values()
    else:
        iterable = getattr(state_dict, "state_dict", lambda: {})().values()

    for value in iterable:
        if torch.is_tensor(value):
            yield value


def estimate_checkpoint_sizes(state_dict, num_bits: int) -> Tuple[float, float]:
    """Estimate raw and quantized storage requirements for tensors in a state dict."""

    original_bytes = 0.0
    quantized_bytes = 0.0

    for tensor in iter_state_tensors(state_dict):
        numel = tensor.numel()
        elem_bytes = tensor.element_size()
        original_bytes += numel * elem_bytes
        if torch.is_floating_point(tensor):
            quantized_bytes += numel * num_bits / 8.0
        else:
            quantized_bytes += numel * elem_bytes

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
) -> None:
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

    original_bytes, quantized_bytes = estimate_checkpoint_sizes(
        state_dict, args.num_bits
    )

    for key, value in state_dict.items():
        if torch.is_tensor(value):
            state_dict[key] = fake_quant_tensor(
                value, args.num_bits, args.quantization
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
