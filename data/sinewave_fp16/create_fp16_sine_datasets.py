#!/usr/bin/env python3
"""Generate multi-context sine-wave datasets encoded as fp16 bit patterns."""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class ContextSpec:
    name: str
    amplitude: float
    phase: float
    period: float
    offset: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", default="data/sinewave_fp16", help="Root directory for generated datasets")
    p.add_argument("--num_contexts", type=int, default=8, help="Number of sine contexts to create")
    p.add_argument("--points_per_period", type=int, default=48)
    p.add_argument("--num_periods", type=int, default=600)
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--base_period", type=float, default=1.0, help="Base frequency multiplier")
    p.add_argument("--period_step", type=float, default=0.15, help="Per-context period increment")
    p.add_argument("--base_amplitude", type=float, default=0.6)
    p.add_argument("--amplitude_step", type=float, default=0.1)
    p.add_argument("--phase_step", type=float, default=0.35, help="Per-context phase offset in radians")
    p.add_argument("--offset", type=float, default=0.0)
    return p.parse_args()


def make_context_specs(args: argparse.Namespace) -> list[ContextSpec]:
    specs: list[ContextSpec] = []
    for i in range(args.num_contexts):
        specs.append(
            ContextSpec(
                name=f"s{i+1}",
                amplitude=args.base_amplitude + i * args.amplitude_step,
                phase=i * args.phase_step,
                period=args.base_period + i * args.period_step,
                offset=args.offset,
            )
        )
    return specs


def generate_fp16_bits(spec: ContextSpec, points_per_period: int, num_periods: int) -> np.ndarray:
    total = points_per_period * num_periods
    x = np.arange(total, dtype=np.float32)
    omega = (2.0 * math.pi) / points_per_period
    y = spec.offset + spec.amplitude * np.sin(omega * spec.period * x + spec.phase)
    y_fp16 = y.astype(np.float16)
    return y_fp16.view(np.uint16)


def write_dataset(output_root: str, spec: ContextSpec, token_bits: np.ndarray, train_split: float, points_per_period: int, num_periods: int) -> None:
    context_dir = os.path.join(output_root, spec.name)
    os.makedirs(context_dir, exist_ok=True)

    split = int(len(token_bits) * train_split)
    train_bits = token_bits[:split]
    val_bits = token_bits[split:]

    train_bits.tofile(os.path.join(context_dir, "train.bin"))
    val_bits.tofile(os.path.join(context_dir, "val.bin"))

    meta = {
        "tokenizer": "sinewave_fp16_bits",
        "vocab_size": 65536,
        "encoding": "fp16_bits",
        "source": "sinewave",
        "sine_period": spec.period,
        "sine_points_per_period": points_per_period,
        "sine_num_periods": num_periods,
        "sine_amplitude": spec.amplitude,
        "sine_phase": spec.phase,
        "sine_offset": spec.offset,
    }
    with open(os.path.join(context_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    args = parse_args()
    specs = make_context_specs(args)
    for spec in specs:
        bits = generate_fp16_bits(spec, args.points_per_period, args.num_periods)
        write_dataset(args.output_root, spec, bits, args.train_split, args.points_per_period, args.num_periods)
    print(f"Created {len(specs)} fp16-bit sinewave datasets under {args.output_root}")
