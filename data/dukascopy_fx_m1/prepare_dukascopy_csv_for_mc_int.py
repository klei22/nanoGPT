#!/usr/bin/env python3
"""Convert Dukascopy M1 CSV(.gz) candles into compact integer columns."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

TICK_COLUMNS = ("open", "high", "low", "close", "volume")
DELTA_STATE_COLUMNS = tuple(f"{column}_delta_state" for column in TICK_COLUMNS)
TIME_COLUMNS = ("minute_mod_10", "minute_of_hour", "minute_of_day", "minute_of_week", "minute_of_year")


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


def percentile(values: Sequence[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def make_delta_thresholds(values: Sequence[int], lower_pct: float, upper_pct: float, minimum_width: float) -> tuple[float, float]:
    negative_magnitudes = [abs(value) for value in values if value < 0]
    positive_values = [value for value in values if value > 0]
    lower = -percentile(negative_magnitudes, upper_pct) if negative_magnitudes else 0.0
    upper = percentile(positive_values, upper_pct) if positive_values else 0.0
    # Fall back to all-value percentiles for degenerate constant-zero columns,
    # then force a non-empty range that still includes zero.
    if lower >= upper:
        lower = min(0.0, percentile(values, lower_pct))
        upper = max(0.0, percentile(values, upper_pct))
    if lower >= upper:
        half_width = max(minimum_width / 2.0, 1.0)
        lower = -half_width
        upper = half_width
    return lower, upper


def signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def encode_delta_state(value: int, lower: float, upper: float, states: int, use_log: bool) -> int:
    clipped = min(max(float(value), lower), upper)
    lo = float(lower)
    hi = float(upper)
    if use_log:
        clipped = signed_log1p(clipped)
        lo = signed_log1p(lo)
        hi = signed_log1p(hi)
    if hi <= lo:
        return states // 2
    scaled = (clipped - lo) / (hi - lo)
    return min(max(round(scaled * (states - 1)), 0), states - 1)


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack("!I", len(payload)) + tag + payload + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def write_histogram_png(path: Path, values: Sequence[int], bins: int = 80, width: int = 960, height: int = 540) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        values = [0]
    v_min = min(values)
    v_max = max(values)
    if v_min == v_max:
        v_min -= 1
        v_max += 1
    bins = max(1, bins)
    counts = [0] * bins
    span = v_max - v_min
    for value in values:
        idx = min(int((value - v_min) / span * bins), bins - 1)
        counts[idx] += 1

    pixels = bytearray([255, 255, 255] * width * height)

    def set_px(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            off = (y * width + x) * 3
            pixels[off : off + 3] = bytes(color)

    margin_l, margin_r, margin_t, margin_b = 72, 24, 36, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    axis = (40, 40, 40)
    bar = (65, 105, 225)
    grid = (225, 225, 225)
    for i in range(6):
        y = margin_t + round(i * plot_h / 5)
        for x in range(margin_l, margin_l + plot_w + 1):
            set_px(x, y, grid)
    for x in range(margin_l, margin_l + plot_w + 1):
        set_px(x, margin_t + plot_h, axis)
    for y in range(margin_t, margin_t + plot_h + 1):
        set_px(margin_l, y, axis)

    max_count = max(counts) or 1
    for idx, count in enumerate(counts):
        x0 = margin_l + round(idx * plot_w / bins)
        x1 = min(margin_l + round((idx + 1) * plot_w / bins), margin_l + plot_w)
        bar_h = round((count / max_count) * (plot_h - 1))
        for x in range(x0, max(x0 + 1, x1 - 1)):
            for y in range(margin_t + plot_h - bar_h, margin_t + plot_h):
                set_px(x, y, bar)

    # Minimal machine-readable labels in sidecar JSON are more useful than
    # bitmap text here; the PNG intentionally remains dependency-free.
    raw_rows = [b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3]) for y in range(height)]
    png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)) + png_chunk(b"IDAT", zlib.compress(b"".join(raw_rows), 9)) + png_chunk(b"IEND", b"")
    path.write_bytes(png)


def time_features(ts: datetime) -> list[int]:
    minute_of_day = ts.hour * 60 + ts.minute
    minute_of_week = ts.weekday() * 24 * 60 + minute_of_day
    minute_of_year = (ts.timetuple().tm_yday - 1) * 24 * 60 + minute_of_day
    return [minute_of_day % 10, ts.minute, minute_of_day, minute_of_week, minute_of_year]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact integer training CSV from Dukascopy candle CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input .csv/.csv.gz files or directories containing them.")
    parser.add_argument("--output_csv", default="data/dukascopy_fx_m1/input.csv", help="Integer CSV to write for data/csv_mc_int/get_dataset.sh.")
    parser.add_argument("--stats-dir", default="data/dukascopy_fx_m1/stats", help="Directory for stats JSON and histogram PNGs.")
    parser.add_argument("--price-scale", type=int, default=100000, help="Multiplier applied to OHLC prices before rounding to integer ticks.")
    parser.add_argument("--volume-scale", type=int, default=1000, help="Multiplier applied to volume before rounding to integer ticks.")
    parser.add_argument("--delta-states", type=int, default=257, help="Number of discrete states for each saturated derivative column.")
    parser.add_argument("--lower-percentile", type=float, default=5.0, help="Lower derivative percentile used as the default negative saturation threshold.")
    parser.add_argument("--upper-percentile", type=float, default=95.0, help="Upper derivative percentile used as the default positive saturation threshold.")
    parser.add_argument("--delta-threshold", action="append", default=[], help="Override a column threshold as <open|high|low|close|volume>:<min>:<max>.")
    parser.add_argument("--log-delta", action="store_true", help="Apply signed log1p preprocessing before bucketing saturated derivative values.")
    args = parser.parse_args()

    if args.delta_states < 2:
        raise ValueError("--delta-states must be at least 2")
    if not (0.0 <= args.lower_percentile < args.upper_percentile <= 100.0):
        raise ValueError("Expected 0 <= --lower-percentile < --upper-percentile <= 100")

    input_paths = iter_input_paths(args.inputs)
    if not input_paths:
        raise ValueError("No input CSV files found")

    manual_thresholds: dict[str, tuple[float, float]] = {}
    for spec in args.delta_threshold:
        parts = spec.split(":")
        if len(parts) != 3 or parts[0] not in TICK_COLUMNS:
            raise ValueError("--delta-threshold must be <open|high|low|close|volume>:<min>:<max>")
        lo, hi = float(parts[1]), float(parts[2])
        if lo >= hi:
            raise ValueError(f"Invalid threshold for {parts[0]}: min must be < max")
        manual_thresholds[parts[0]] = (lo, hi)

    rows: list[dict[str, object]] = []
    raw_ticks: dict[str, list[int]] = {column: [] for column in TICK_COLUMNS}
    for input_path in input_paths:
        with open_text(input_path) as in_f:
            reader = csv.DictReader(in_f)
            missing = {"timestamp_utc", *TICK_COLUMNS}.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{input_path} missing required columns: {sorted(missing)}")
            for row in reader:
                ticks = {column: round(float(row[column]) * (args.volume_scale if column == "volume" else args.price_scale)) for column in TICK_COLUMNS}
                ts = parse_timestamp_utc(row["timestamp_utc"])
                rows.append({"ts": ts, "ticks": ticks})
                for column, value in ticks.items():
                    raw_ticks[column].append(value)

    if len(rows) < 2:
        raise ValueError(f"Need at least two candle rows for train/val split; found {len(rows)}")

    deltas: dict[str, list[int]] = {column: [] for column in TICK_COLUMNS}
    previous: dict[str, int] | None = None
    for row in rows:
        ticks = row["ticks"]
        assert isinstance(ticks, dict)
        for column in TICK_COLUMNS:
            current = int(ticks[column])
            deltas[column].append(0 if previous is None else current - previous[column])
        previous = {column: int(ticks[column]) for column in TICK_COLUMNS}

    thresholds: dict[str, tuple[float, float]] = {}
    stats_dir = Path(args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": len(rows),
        "delta_states": args.delta_states,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "log_delta": bool(args.log_delta),
        "columns": {},
    }
    for column in TICK_COLUMNS:
        thresholds[column] = manual_thresholds.get(column) or make_delta_thresholds(deltas[column], args.lower_percentile, args.upper_percentile, 1.0)
        lower, upper = thresholds[column]
        stats["columns"][column] = {
            "raw_ticks_min": min(raw_ticks[column]),
            "raw_ticks_max": max(raw_ticks[column]),
            "delta_min": min(deltas[column]),
            "delta_max": max(deltas[column]),
            "delta_saturation_min": lower,
            "delta_saturation_max": upper,
            "raw_histogram_png": str(stats_dir / f"{column}_ticks_hist.png"),
            "delta_histogram_png": str(stats_dir / f"{column}_delta_hist.png"),
        }
        write_histogram_png(stats_dir / f"{column}_ticks_hist.png", raw_ticks[column])
        write_histogram_png(stats_dir / f"{column}_delta_hist.png", deltas[column])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [*TIME_COLUMNS, *DELTA_STATE_COLUMNS]
    with output_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        for row_idx, row in enumerate(rows):
            ts = row["ts"]
            assert isinstance(ts, datetime)
            out_row = time_features(ts)
            for column in TICK_COLUMNS:
                lower, upper = thresholds[column]
                out_row.append(encode_delta_state(deltas[column][row_idx], lower, upper, args.delta_states, args.log_delta))
            writer.writerow(out_row)

    with (stats_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote derivative stats and histograms to {stats_dir}")


if __name__ == "__main__":
    main()
