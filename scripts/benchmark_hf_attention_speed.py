#!/usr/bin/env python3
"""Benchmark matched ReLU2Max and softmax attention training iterations on CUDA."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn import functional as F

from hf_model.triton_relu2max import TRITON_AVAILABLE, triton_relu2max


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--qk-scale", type=float, default=19.99859042974533,
                        help="Learned QK scale; default is log2(1024^2-1024)")
    parser.add_argument("--relu2max-divisor", type=float, default=256.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output", default="attention_speed_comparison.txt")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def attention_step(q, k, v, causal_mask, variant, qk_scale, divisor):
    if variant == "softmax_sdpa":
        # This is the conventional PyTorch path and allows SDPA to select Flash
        # Attention on an A100. Passing scale preserves the learned QK factor.
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True, scale=qk_scale
        )

    scores = (q @ k.transpose(-2, -1)) * qk_scale
    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    if variant == "softmax_generic":
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
    elif variant == "relu2max_torch":
        weights = F.relu(scores).square() / divisor
    elif variant == "relu2max_triton":
        weights = triton_relu2max(scores, divisor)
    else:
        raise ValueError(f"unknown benchmark variant: {variant}")
    return weights @ v


def benchmark_variant(args, variant, q, k, v, causal_mask):
    def iteration():
        output = attention_step(
            q, k, v, causal_mask, variant, args.qk_scale, args.relu2max_divisor
        )
        # Backpropagate through QK, normalization, and PV, as in training.
        output.float().square().mean().backward()
        q.grad = k.grad = v.grad = None

    for _ in range(args.warmup_iterations):
        iteration()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iterations):
        iteration()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    average_ms = elapsed_ms / args.iterations
    return {
        "variant": variant,
        "average_ms": average_ms,
        "iterations_per_second": 1000.0 / average_ms,
        "tokens_per_second": args.batch_size * args.sequence_length * 1000.0 / average_ms,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def format_table(results):
    headers = ("Variation", "Avg iteration (ms)", "Iterations/s", "Tokens/s", "Peak GiB", "vs generic")
    generic_ms = next(row["average_ms"] for row in results if row["variant"] == "softmax_generic")
    rows = []
    labels = {
        "relu2max_torch": "ReLU2Max (PyTorch)",
        "relu2max_triton": "ReLU2Max (Triton)",
        "softmax_generic": "Softmax (generic)",
        "softmax_sdpa": "Softmax (SDPA/conventional)",
    }
    for result in results:
        rows.append((
            labels[result["variant"]],
            f'{result["average_ms"]:.3f}',
            f'{result["iterations_per_second"]:.2f}',
            f'{result["tokens_per_second"]:,.0f}',
            f'{result["peak_memory_gib"]:.2f}',
            f'{generic_ms / result["average_ms"]:.2f}x',
        ))
    widths = [max(len(str(value)) for value in column) for column in zip(headers, *rows)]
    render = lambda row: " | ".join(str(value).ljust(width) for value, width in zip(row, widths))
    separator = "-+-".join("-" * width for width in widths)
    return "\n".join((render(headers), separator, *(render(row) for row in rows)))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")
    if not TRITON_AVAILABLE:
        raise RuntimeError("This benchmark includes ReLU2Max Triton; install Triton first")
    if args.iterations <= 0 or args.warmup_iterations < 0:
        raise ValueError("iterations must be positive and warmup iterations non-negative")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    shape = (args.batch_size, args.heads, args.sequence_length, args.head_dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q, k = q.detach().requires_grad_(True), k.detach().requires_grad_(True)
    causal_mask = torch.ones(
        (1, 1, args.sequence_length, args.sequence_length), device="cuda", dtype=torch.bool
    ).tril_()

    variants = ("relu2max_torch", "relu2max_triton", "softmax_generic", "softmax_sdpa")
    results = [benchmark_variant(args, variant, q, k, v, causal_mask) for variant in variants]
    table = format_table(results)
    metadata = (
        f"device={torch.cuda.get_device_name()} dtype={args.dtype} shape={shape} "
        f"warmup={args.warmup_iterations} measured_iterations={args.iterations}\n"
        f"one_materialized_score_tensor={args.batch_size * args.heads * args.sequence_length**2 * q.element_size() / 1024**3:.2f} GiB "
        "(generic paths create/read additional score, mask, and gradient intermediates)\n"
    )
    report = metadata + table + "\n"
    print(report, end="")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"\nSaved comparison to {output}")


if __name__ == "__main__":
    main()
