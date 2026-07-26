#!/usr/bin/env python3
"""Convert a Whisper mel CSV (no header) into a headered CSV for int multicontext demos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_bin_selection(raw: str, n_mels: int) -> list[int]:
    bins: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= n_mels:
            raise ValueError(f"bin index {idx} outside [0, {n_mels - 1}]")
        bins.append(idx)
    if not bins:
        raise ValueError("No bins selected")
    return bins


def read_mel_csv(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_idx, row in enumerate(reader, start=1):
            if not row:
                continue
            try:
                rows.append([float(v) for v in row])
            except ValueError as exc:
                raise ValueError(f"Non-float value at line {line_idx} in {path}") from exc
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    n_cols = len(rows[0])
    for idx, row in enumerate(rows, start=1):
        if len(row) != n_cols:
            raise ValueError(f"Inconsistent column count at line {idx}: expected {n_cols}, got {len(row)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select mel bins and emit a headered CSV.")
    parser.add_argument("--input_csv", required=True, help="Path to mel CSV (frames x mel_bins).")
    parser.add_argument("--output_csv", default="audio_num_int_input.csv", help="Output CSV path with header.")
    parser.add_argument(
        "--bins",
        default="10,30,60",
        help="Comma-separated mel bin indices to keep (defaults to 3 channels for multicontext demos).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    rows = read_mel_csv(input_path)
    n_mels = len(rows[0])
    bins = parse_bin_selection(args.bins, n_mels)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [f"mel_bin_{idx:03d}" for idx in bins]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([f"{row[idx]:.6f}" for idx in bins])

    print(
        f"Converted {input_path} ({len(rows)} frames x {n_mels} bins) -> "
        f"{output_path} using bins {bins}"
    )


if __name__ == "__main__":
    main()
