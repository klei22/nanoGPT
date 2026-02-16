#!/usr/bin/env python3
"""Create fp16-bit encoded sine-wave datasets for numerical multicontext training."""

import argparse
import math
import os
import pickle
from typing import List

import numpy as np


def _parse_csv_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _float_to_fp16_bits(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).view(np.uint16)


def build_wave(period: float, phase: float, amplitude: float, offset: float, total_points: int) -> np.ndarray:
    x = np.arange(total_points, dtype=np.float32)
    angle = ((x * 2.0 * math.pi) / period) + phase
    return offset + amplitude * np.sin(angle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fp16-bit sine datasets in data/<dataset>/s*/")
    parser.add_argument("--dataset_root", default="data/sinewave_fp16", help="Root output folder")
    parser.add_argument("--num_contexts", type=int, default=8, help="Number of contexts/datasets to create")
    parser.add_argument("--points_per_context", type=int, default=32768, help="Total samples per context")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train split fraction")
    parser.add_argument("--period", type=float, default=64.0, help="Sine period in points")
    parser.add_argument("--offset", type=float, default=0.0, help="DC offset")
    parser.add_argument("--phases", type=str, default="0.0,0.785398,1.570796,2.356194,3.141593,3.926991,4.712389,5.497787")
    parser.add_argument("--amplitudes", type=str, default="0.25,0.5,0.75,1.0,1.25,0.9,0.6,0.35")
    args = parser.parse_args()

    phases = _parse_csv_floats(args.phases)
    amplitudes = _parse_csv_floats(args.amplitudes)

    if len(phases) < args.num_contexts or len(amplitudes) < args.num_contexts:
        raise ValueError("Provide at least num_contexts entries for --phases and --amplitudes")

    os.makedirs(args.dataset_root, exist_ok=True)

    train_len = int(args.points_per_context * args.train_split)
    val_len = args.points_per_context - train_len

    for i in range(args.num_contexts):
        context_name = f"s{i+1}"
        context_dir = os.path.join(args.dataset_root, context_name)
        os.makedirs(context_dir, exist_ok=True)

        wave = build_wave(
            period=args.period,
            phase=phases[i],
            amplitude=amplitudes[i],
            offset=args.offset,
            total_points=args.points_per_context,
        ).astype(np.float32)
        wave_bits = _float_to_fp16_bits(wave)

        train_bits = wave_bits[:train_len]
        val_bits = wave_bits[train_len:]

        train_bits.tofile(os.path.join(context_dir, "train.bin"))
        val_bits.tofile(os.path.join(context_dir, "val.bin"))

        meta = {
            "tokenizer": "sinewave",
            "sine_encoding": "fp16_bits",
            "vocab_size": 65536,
            "period": args.period,
            "phase": phases[i],
            "amplitude": amplitudes[i],
            "offset": args.offset,
            "train_samples": train_len,
            "val_samples": val_len,
        }
        with open(os.path.join(context_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        print(f"Wrote {context_dir}: phase={phases[i]:.6f}, amplitude={amplitudes[i]:.3f}")


if __name__ == "__main__":
    main()
