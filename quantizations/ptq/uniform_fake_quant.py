import argparse
import os
import shutil
import torch
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply uniform fake quantization to all weights in a checkpoint",
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
        help="Number of bits to use for fake quantization",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview stats but DO NOT write checkpoint",
    )
    return parser.parse_args()


console = Console()


def fake_quantize_tensor(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Quantize ``tensor`` to ``num_bits`` using symmetric uniform quantization."""
    if tensor.ndim == 0:
        return tensor
    qmax = 2 ** (num_bits - 1) - 1
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = max_val / qmax
    q = torch.clamp((tensor / scale).round(), -qmax, qmax)
    return q * scale


def main():
    args = parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    meta_path = os.path.join(args.ckpt_dir, "meta.pkl")

    console.rule("[bold cyan]Loading checkpoint")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)

    # optimizer and scheduler states depend on parameter precision.
    # Drop them as they are invalid after quantization.
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scheduler", None)

    total_fp = sum(1 for t in state_dict.values() if torch.is_floating_point(t))
    console.rule("[bold cyan]Quantizing tensors")
    bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    )
    tid = bar.add_task("PTQ", total=total_fp)

    with bar:
        for key, tensor in list(state_dict.items()):
            if not torch.is_floating_point(tensor):
                continue
            state_dict[key] = fake_quantize_tensor(tensor, args.num_bits)
            bar.advance(tid)

    if args.dry_run:
        console.rule("[bold red]Dry-run complete -- nothing written[/]")
        console.print(
            f"Visited {total_fp} tensors; num_bits={args.num_bits}",
        )
        return

    out_dir = args.out_dir or f"{args.ckpt_dir.rstrip('/').rstrip(os.sep)}_ptq"
    os.makedirs(out_dir, exist_ok=True)

    console.rule("[bold cyan]Saving quantized checkpoint")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(out_dir, "meta.pkl"))

    console.print(
        f"[green]✔ All done[/] → {out_dir}  "
        f"([bold]{args.num_bits}[/]-bit fake quant)",
    )


if __name__ == "__main__":
    main()

