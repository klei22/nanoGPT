#!/usr/bin/env python3
"""Preprocess minimum-angle graph CSV exports for the fast browser viewer.

The training/export path stays unchanged: this script reads the existing CSV
format emitted by utils/min_angle_graph_export.py and writes compact columnar
JSON snapshots for analysis/min_angle_graph_fast_viewer.html.
"""

import argparse
import csv
import json
import math
from pathlib import Path

NUMERIC_COLUMNS = {
    "token_id": int,
    "nearest_token_id": int,
    "min_angle_deg": float,
    "cosine": float,
    "token_vector_length": float,
    "nearest_token_vector_length": float,
    "min_angle_rank": int,
}
TEXT_COLUMNS = ("token_text_escaped", "nearest_token_text_escaped")
REQUIRED_COLUMNS = tuple(NUMERIC_COLUMNS) + TEXT_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing original minimum-angle graph CSV files.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Destination folder. Defaults to <input_dir>/fast_viewer.")
    parser.add_argument("--glob", default="*.csv", help="CSV glob relative to input_dir. Default: *.csv")
    parser.add_argument("--histogram_bins", type=int, default=120, help="Number of precomputed angle histogram bins.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON for debugging at the cost of larger files.")
    return parser.parse_args()


def finite_float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sidecar_metadata(csv_path):
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_histogram(values, bins):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"min": None, "max": None, "bin_width": None, "centers": [], "counts": []}
    lo, hi = min(finite), max(finite)
    if lo == hi:
        return {"min": lo, "max": hi, "bin_width": 1.0, "centers": [lo], "counts": [len(finite)]}
    bins = max(1, int(bins))
    width = (hi - lo) / bins
    counts = [0] * bins
    for value in finite:
        index = min(bins - 1, int((value - lo) / width))
        counts[index] += 1
    centers = [lo + (index + 0.5) * width for index in range(bins)]
    return {"min": lo, "max": hi, "bin_width": width, "centers": centers, "counts": counts}


def read_csv_columns(csv_path):
    columns = {column: [] for column in REQUIRED_COLUMNS}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in NUMERIC_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        for row in reader:
            for column, converter in NUMERIC_COLUMNS.items():
                if converter is int:
                    columns[column].append(finite_int(row.get(column)))
                else:
                    columns[column].append(finite_float(row.get(column)))
            for column in TEXT_COLUMNS:
                columns[column].append(row.get(column, ""))
    return columns


def preprocess_csv(csv_path, output_dir, histogram_bins, pretty=False):
    columns = read_csv_columns(csv_path)
    row_count = len(columns["token_id"])
    all_indices = list(range(row_count))
    indices = {
        "by_rank": sorted(all_indices, key=lambda idx: columns["min_angle_rank"][idx]),
        "by_token": sorted(all_indices, key=lambda idx: columns["token_id"][idx]),
        "by_angle_desc": sorted(all_indices, key=lambda idx: columns["min_angle_deg"][idx], reverse=True),
    }
    metadata = sidecar_metadata(csv_path)
    metadata.update({
        "source_csv": str(csv_path),
        "optimized_for": "analysis/min_angle_graph_fast_viewer.html",
        "format_version": 1,
    })
    payload = {
        "metadata": metadata,
        "row_count": row_count,
        "columns": columns,
        "indices": indices,
        "histogram": build_histogram(columns["min_angle_deg"], histogram_bins),
    }
    output_path = output_dir / f"{csv_path.stem}.min-angle-graph.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2 if pretty else None, separators=None if pretty else (",", ":"))
    return output_path, row_count


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or (input_dir / "fast_viewer")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not csv_paths:
        raise SystemExit(f"No CSV files matched {args.glob!r} in {input_dir}")
    manifest = {"format_version": 1, "source_dir": str(input_dir), "snapshots": []}
    for csv_path in csv_paths:
        output_path, row_count = preprocess_csv(csv_path, output_dir, args.histogram_bins, pretty=args.pretty)
        manifest["snapshots"].append({"source_csv": str(csv_path), "optimized_json": output_path.name, "row_count": row_count})
        print(f"wrote {output_path} ({row_count} rows)")
    manifest_path = output_dir / "manifest.min-angle-graph.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2 if args.pretty else None)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
