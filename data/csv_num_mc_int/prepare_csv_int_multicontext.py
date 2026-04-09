#!/usr/bin/env python3
"""Create integer-quantized numerical multicontext datasets from CSV columns.

Pipeline per value:
    transformed = (raw + shift) * scale
    quantized_int = round(transformed)

Each column becomes one context directory under data/<output_root>/.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


UINT16_MIN = 0
UINT16_MAX = 65535


def parse_column_transform(spec: str) -> Tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --column-transform '{spec}'. Expected <column>:<shift>:<scale>")
    column, shift_str, scale_str = parts
    if not column:
        raise ValueError(f"Invalid --column-transform '{spec}': empty column")
    return column, float(shift_str), float(scale_str)


def clean_context_name(header: str, fallback_idx: int) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in header.strip()).strip("_")
    return cleaned or f"column_{fallback_idx + 1}"


def read_csv_columns(input_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError("CSV must contain a header row")

        data: Dict[str, List[float]] = {h: [] for h in headers}
        for row_idx, row in enumerate(reader, start=2):
            for header in headers:
                cell = (row.get(header) or "").strip()
                if cell == "":
                    raise ValueError(f"Empty cell at row {row_idx}, column '{header}'")
                data[header].append(float(cell))

    total = len(data[headers[0]])
    if total < 2:
        raise ValueError("CSV must contain at least 2 rows")
    if any(len(col) != total for col in data.values()):
        raise ValueError("All columns must contain the same number of rows")
    return headers, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build int-quantized multicontext datasets from CSV numeric columns.")
    parser.add_argument("--input_csv", default="input.csv", help="Path to input CSV (header required).")
    parser.add_argument("--output_root", default="csv_num_mc_int", help="Output folder under data/.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio in (0,1).")
    parser.add_argument("--column-transform", action="append", default=[], help="Repeatable '<column>:<shift>:<scale>'.")
    parser.add_argument("--clip_min", type=int, default=UINT16_MIN, help="Minimum integer value after quantization.")
    parser.add_argument("--clip_max", type=int, default=UINT16_MAX, help="Maximum integer value after quantization.")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")
    if args.clip_min < UINT16_MIN or args.clip_max > UINT16_MAX or args.clip_min > args.clip_max:
        raise ValueError(f"clip range must be within [0,{UINT16_MAX}] and clip_min<=clip_max")

    transforms: Dict[str, Tuple[float, float]] = {}
    for item in args.column_transform:
        column, shift, scale = parse_column_transform(item)
        transforms[column] = (shift, scale)

    input_path = Path(args.input_csv).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    headers, data = read_csv_columns(input_path)
    total_samples = len(data[headers[0]])
    train_n = int(total_samples * args.train_ratio)
    if train_n <= 0 or train_n >= total_samples:
        raise ValueError(f"Invalid split: train={train_n}, total={total_samples}")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / "data" / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output root: {output_root}")
    print(f"Rows/column: total={total_samples}, train={train_n}, val={total_samples - train_n}")

    for idx, header in enumerate(headers):
        raw = np.asarray(data[header], dtype=np.float32)
        shift, scale = transforms.get(header, (0.0, 1.0))

        transformed = (raw + shift) * scale
        rounded = np.rint(transformed)
        quantized = np.clip(rounded, args.clip_min, args.clip_max).astype(np.uint16)

        context_name = clean_context_name(header, idx)
        context_dir = output_root / context_name
        context_dir.mkdir(parents=True, exist_ok=True)

        quantized[:train_n].tofile(context_dir / "train.bin")
        quantized[train_n:].tofile(context_dir / "val.bin")

        meta = {
            "tokenizer": "csv_quantized_int",
            "vocab_size": UINT16_MAX + 1,
            "numerical_multicontext_input_format": "scalar",
            "source_csv": str(input_path),
            "source_column": header,
            "samples": int(total_samples),
            "train_ratio": float(args.train_ratio),
            "quantization": {
                "type": "shift_scale_round_clip_uint16",
                "shift": float(shift),
                "scale": float(scale),
                "clip_min": int(args.clip_min),
                "clip_max": int(args.clip_max),
                "inverse": "raw ~= (quantized / scale) - shift (for scale != 0)",
            },
            "float_range_raw_min": float(raw.min()),
            "float_range_raw_max": float(raw.max()),
            "float_range_transformed_min": float(transformed.min()),
            "float_range_transformed_max": float(transformed.max()),
            "int_range_quantized_min": int(quantized.min()),
            "int_range_quantized_max": int(quantized.max()),
        }
        with (context_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        print(
            f"[{context_name}] column='{header}', shift={shift}, scale={scale}, "
            f"raw[min,max]=({raw.min():.6g},{raw.max():.6g}), "
            f"tx[min,max]=({transformed.min():.6g},{transformed.max():.6g}), "
            f"q[min,max]=({quantized.min()},{quantized.max()})"
        )


if __name__ == "__main__":
    main()
