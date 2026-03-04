#!/usr/bin/env python3
"""Create fp16-bit numerical multicontext datasets from CSV columns."""

from __future__ import annotations

import argparse
import csv
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Tuple


def parse_column_transform(spec: str) -> Tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --column-transform '{spec}'. Expected <column>:<offset>:<scale>")
    column, offset_str, scale_str = parts
    if not column:
        raise ValueError(f"Invalid --column-transform '{spec}': empty column")
    return column, float(offset_str), float(scale_str)


def clean_context_name(header: str, fallback_idx: int) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in header.strip()).strip("_")
    return cleaned or f"column_{fallback_idx + 1}"


def fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def write_u16_file(path: Path, values: List[int]) -> None:
    with path.open("wb") as f:
        for v in values:
            f.write(struct.pack("<H", v))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fp16-bit multicontext datasets from CSV numeric columns.")
    parser.add_argument("--input_csv", default="input.csv", help="Path to input CSV (header required).")
    parser.add_argument("--output_root", default="csv_num_mc", help="Output folder under data/.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio in (0,1).")
    parser.add_argument("--column-transform", action="append", default=[], help="Repeatable '<column>:<offset>:<scale>'.")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")

    transforms: Dict[str, Tuple[float, float]] = {}
    for item in args.column_transform:
        column, offset, scale = parse_column_transform(item)
        transforms[column] = (offset, scale)

    input_path = Path(args.input_csv).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

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

    total_samples = len(data[headers[0]])
    if total_samples < 2:
        raise ValueError("CSV must contain at least 2 rows")
    if any(len(v) != total_samples for v in data.values()):
        raise ValueError("All columns must contain the same number of rows")

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
        context_name = clean_context_name(header, idx)
        context_dir = output_root / context_name
        context_dir.mkdir(parents=True, exist_ok=True)

        raw = data[header]
        offset, scale = transforms.get(header, (0.0, 1.0))
        transformed = [(v + offset) * scale for v in raw]
        bit_values = [fp16_bits(v) for v in transformed]

        write_u16_file(context_dir / "train.bin", bit_values[:train_n])
        write_u16_file(context_dir / "val.bin", bit_values[train_n:])

        meta = {
            "tokenizer": "csv_fp16_bits",
            "encoding": "ieee754-fp16-bitpattern-in-uint16",
            "vocab_size": 65536,
            "numerical_multicontext_input_format": "fp16_bits",
            "source_csv": str(input_path),
            "source_column": header,
            "samples": total_samples,
            "train_ratio": args.train_ratio,
            "applied_offset": offset,
            "applied_scale": scale,
            "float_range_raw_min": min(raw),
            "float_range_raw_max": max(raw),
            "float_range_transformed_min": min(transformed),
            "float_range_transformed_max": max(transformed),
        }
        with (context_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        print(
            f"[{context_name}] column='{header}', offset={offset}, scale={scale}, "
            f"raw[min,max]=({min(raw):.6g},{max(raw):.6g}), "
            f"tx[min,max]=({min(transformed):.6g},{max(transformed):.6g})"
        )


if __name__ == "__main__":
    main()
