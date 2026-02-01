#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt


TRAIN_COL_INDEX = 2
VAL_COL_INDEX = 3
ITER_COL_INDEX = 0
TOKENS_COL_INDEX = 9


@dataclass(frozen=True)
class CsvSeries:
    name: str
    x_values: list[float]
    train_values: list[float]
    val_values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live viewer for bulk CSV logs. Plots train/val loss from each CSV "
            "file on a single graph and refreshes when files update."
        )
    )
    parser.add_argument(
        "--csv-dir",
        default="csv_logs",
        help="Directory containing CSV logs (default: csv_logs).",
    )
    parser.add_argument(
        "--pattern",
        default="bulk_*.csv",
        help="Glob pattern for CSV files (default: bulk_*.csv).",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=5.0,
        help="How often to check for file updates (default: 5 seconds).",
    )
    parser.add_argument(
        "--x-axis",
        choices=("iter", "tokens"),
        default="iter",
        help="X-axis column to plot (iter or tokens).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum number of points to display (0 for all).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render once and exit (no live updates).",
    )
    return parser.parse_args()


def find_csv_files(csv_dir: str, pattern: str) -> list[str]:
    search_pattern = os.path.join(csv_dir, pattern)
    return sorted(glob.glob(search_pattern))


def load_csv_series(path: str, x_col_index: int, max_points: int) -> CsvSeries:
    x_values: list[float] = []
    train_values: list[float] = []
    val_values: list[float] = []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) <= max(TRAIN_COL_INDEX, VAL_COL_INDEX, x_col_index):
                continue
            try:
                x_value = float(row[x_col_index])
                train_value = float(row[TRAIN_COL_INDEX])
                val_value = float(row[VAL_COL_INDEX])
            except ValueError:
                continue
            x_values.append(x_value)
            train_values.append(train_value)
            val_values.append(val_value)
    if max_points > 0 and len(x_values) > max_points:
        x_values = x_values[-max_points:]
        train_values = train_values[-max_points:]
        val_values = val_values[-max_points:]
    name = os.path.basename(path)
    return CsvSeries(name=name, x_values=x_values, train_values=train_values, val_values=val_values)


def collect_series(paths: list[str], x_col_index: int, max_points: int) -> list[CsvSeries]:
    return [load_csv_series(path, x_col_index, max_points) for path in paths]


def plot_series(series_list: list[CsvSeries], x_axis_label: str) -> None:
    plt.cla()
    if not series_list:
        plt.title("CSV Logs (no matching files found)")
        plt.xlabel(x_axis_label)
        plt.ylabel("loss")
        plt.pause(0.1)
        return
    for series in series_list:
        if not series.x_values:
            continue
        plt.plot(series.x_values, series.train_values, label=f"{series.name} train")
        plt.plot(series.x_values, series.val_values, label=f"{series.name} val")
    plt.title("CSV Logs (train/val loss)")
    plt.xlabel(x_axis_label)
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small")
    plt.tight_layout()


def get_file_signatures(paths: list[str]) -> dict[str, tuple[float, int]]:
    signatures: dict[str, tuple[float, int]] = {}
    for path in paths:
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            continue
        signatures[path] = (stat.st_mtime, stat.st_size)
    return signatures


def main() -> None:
    args = parse_args()
    x_col_index = ITER_COL_INDEX if args.x_axis == "iter" else TOKENS_COL_INDEX

    plt.ion()
    fig, _ = plt.subplots()
    fig.canvas.manager.set_window_title("nanoGPT CSV Log Viewer")

    last_signatures: dict[str, tuple[float, int]] = {}
    try:
        while True:
            paths = find_csv_files(args.csv_dir, args.pattern)
            signatures = get_file_signatures(paths)
            if signatures != last_signatures or args.once:
                series_list = collect_series(paths, x_col_index, args.max_points)
                plot_series(series_list, args.x_axis)
                last_signatures = signatures
                plt.pause(0.1)
            if args.once:
                break
            time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
