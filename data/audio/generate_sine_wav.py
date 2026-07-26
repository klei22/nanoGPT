#!/usr/bin/env python3
"""Generate a mono sine-wave WAV file for toy audio experiments."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sine-wave WAV file.")
    parser.add_argument("--output", type=Path, default=Path("dummy_sine.wav"), help="Output WAV path.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate in Hz.")
    parser.add_argument("--frequency", type=float, default=440.0, help="Sine frequency in Hz.")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds.")
    parser.add_argument("--amplitude", type=float, default=0.25, help="Amplitude in [0, 1].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0:
        raise ValueError("--sample_rate must be > 0")
    if args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if not (0.0 < args.amplitude <= 1.0):
        raise ValueError("--amplitude must be in (0, 1]")

    n_samples = int(math.floor(args.sample_rate * args.duration))
    frames = bytearray()
    for i in range(n_samples):
        t = i / float(args.sample_rate)
        sample = args.amplitude * math.sin(2.0 * math.pi * args.frequency * t)
        pcm = int(max(-1.0, min(1.0, sample)) * 32767)
        frames.extend(int(pcm).to_bytes(2, byteorder="little", signed=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(args.sample_rate)
        wf.writeframes(bytes(frames))

    print(f"Wrote {args.output} ({n_samples} samples @ {args.sample_rate} Hz)")


if __name__ == "__main__":
    main()
