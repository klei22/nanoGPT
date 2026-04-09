#!/usr/bin/env python3
"""Create multi-context sine-wave datasets encoded as float16 bit patterns.

Each context writes:
  data/<output_root>/<context>/train.bin (uint16)
  data/<output_root>/<context>/val.bin   (uint16)
  data/<output_root>/<context>/meta.pkl

The uint16 values are raw IEEE-754 fp16 bit patterns. Use
`--numerical_multicontext_input_format fp16_bits` at train/inference time.

Generation is aligned to the regular sinewave data generator:
- x_i = (i * 2π) / points_per_period
- y_i = dc_offset + amplitude * sin(x_i * period + phase)
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np


def _make_wave(
    *,
    period: float,
    phase: float,
    amplitude: float,
    dc_offset: float,
    points_per_period: int,
    num_periods: int,
) -> np.ndarray:
    total_points = points_per_period * num_periods
    idx = np.arange(total_points, dtype=np.float32)
    radians = (idx * 2.0 * math.pi) / float(points_per_period)
    values = dc_offset + amplitude * np.sin(radians * period + phase)
    return values.astype(np.float32)


def _fp32_to_fp16_bits(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).view(np.uint16)


def _write_context(output_root: Path, context_name: str, train_bits: np.ndarray, val_bits: np.ndarray, metadata: dict) -> None:
    context_dir = output_root / context_name
    context_dir.mkdir(parents=True, exist_ok=True)
    train_bits.astype(np.uint16).tofile(context_dir / "train.bin")
    val_bits.astype(np.uint16).tofile(context_dir / "val.bin")
    with (context_dir / "meta.pkl").open("wb") as f:
        pickle.dump(metadata, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fp16-bit sinewave datasets for numerical multicontext training.")
    parser.add_argument("--output_root", default="sinewave_fp16", help="Output directory under data/.")
    parser.add_argument("--contexts", type=int, default=8, help="Number of contexts to generate.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio.")

    # Regular sinewave-like controls
    parser.add_argument("--base_period", type=float, default=15.0, help="Base sine period multiplier for context 1.")
    parser.add_argument("--period_step", type=float, default=1.0, help="Period increment per context.")
    parser.add_argument("--points_per_period", type=int, default=15, help="Discrete points sampled per period.")
    parser.add_argument("--num_periods", type=int, default=2000, help="Number of periods to generate.")

    parser.add_argument("--base_phase", type=float, default=0.0, help="Base phase in radians for context 1.")
    parser.add_argument("--phase_step", type=float, default=0.0, help="Phase increment per context (default 0 to match regular).")

    parser.add_argument("--base_amplitude", type=float, default=50.0, help="Base amplitude for context 1.")
    parser.add_argument("--amplitude_step", type=float, default=0.0, help="Amplitude increment per context (default 0 to match regular).")
    parser.add_argument("--dc_offset", type=float, default=64.0, help="DC offset (regular sinewave uses 64).")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")

    total_samples = args.points_per_period * args.num_periods
    train_n = int(total_samples * args.train_ratio)
    val_n = total_samples - train_n

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / "data" / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Writing {args.contexts} contexts into: {output_root}")
    print(f"Samples/context: total={total_samples}, train={train_n}, val={val_n}")

    for idx in range(args.contexts):
        period = args.base_period + idx * args.period_step
        phase = args.base_phase + idx * args.phase_step
        amplitude = args.base_amplitude + idx * args.amplitude_step
        context_name = f"s{idx + 1}"

        wave = _make_wave(
            period=period,
            phase=phase,
            amplitude=amplitude,
            dc_offset=args.dc_offset,
            points_per_period=args.points_per_period,
            num_periods=args.num_periods,
        )
        bits = _fp32_to_fp16_bits(wave)
        train_bits = bits[:train_n]
        val_bits = bits[train_n:]

        metadata = {
            "tokenizer": "sinewave_fp16_bits",
            "encoding": "ieee754-fp16-bitpattern-in-uint16",
            "vocab_size": 65536,
            "numerical_multicontext_input_format": "fp16_bits",
            "points_per_period": args.points_per_period,
            "num_periods": args.num_periods,
            "samples": total_samples,
            "train_ratio": args.train_ratio,
            "period": period,
            "phase_radians": phase,
            "amplitude": amplitude,
            "dc_offset": args.dc_offset,
            "float_range_min": float(wave.min()),
            "float_range_max": float(wave.max()),
        }

        _write_context(output_root, context_name, train_bits, val_bits, metadata)
        print(
            f"[{context_name}] period={period:.4f}, phase={phase:.4f}, amplitude={amplitude:.4f}, "
            f"float[min,max]=({wave.min():.4f}, {wave.max():.4f})"
        )


if __name__ == "__main__":
    main()
