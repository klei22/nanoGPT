#!/usr/bin/env python3
"""Convert Dukascopy M1 CSV(.gz) candles into integer columns for csv_mc_int."""

from __future__ import annotations

import argparse
import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PRICE_COLUMNS = ("open", "high", "low", "close")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def parse_timestamp_utc(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iter_input_paths(paths: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.csv")))
            resolved.extend(sorted(path.rglob("*.csv.gz")))
        else:
            resolved.append(path)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an integer training CSV from Dukascopy candle CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input .csv/.csv.gz files or directories containing them.")
    parser.add_argument("--output_csv", default="data/dukascopy_fx_m1/input.csv", help="Integer CSV to write for data/csv_mc_int/get_dataset.sh.")
    parser.add_argument("--price-scale", type=int, default=100000, help="Multiplier applied to OHLC prices before rounding to integer ticks.")
    parser.add_argument("--volume-scale", type=int, default=1000, help="Multiplier applied to volume before rounding to integer ticks.")
    parser.add_argument("--include-weekday", action="store_true", help="Include UTC weekday column (0=Monday ... 6=Sunday).")
    args = parser.parse_args()

    input_paths = iter_input_paths(args.inputs)
    if not input_paths:
        raise ValueError("No input CSV files found")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["minute_of_day"]
    if args.include_weekday:
        header.append("weekday")
    header.extend(["open_ticks", "high_ticks", "low_ticks", "close_ticks", "volume_ticks"])

    rows_written = 0
    with output_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        for input_path in input_paths:
            with open_text(input_path) as in_f:
                reader = csv.DictReader(in_f)
                missing = {"timestamp_utc", "volume", *PRICE_COLUMNS}.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{input_path} missing required columns: {sorted(missing)}")
                for row in reader:
                    ts = parse_timestamp_utc(row["timestamp_utc"])
                    out_row = [ts.hour * 60 + ts.minute]
                    if args.include_weekday:
                        out_row.append(ts.weekday())
                    for column in PRICE_COLUMNS:
                        out_row.append(round(float(row[column]) * args.price_scale))
                    out_row.append(round(float(row["volume"]) * args.volume_scale))
                    writer.writerow(out_row)
                    rows_written += 1

    if rows_written < 2:
        raise ValueError(f"Need at least two candle rows for train/val split; wrote {rows_written}")
    print(f"Wrote {rows_written} rows to {output_csv}")


if __name__ == "__main__":
    main()
