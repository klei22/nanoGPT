#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass
from shutil import get_terminal_size
from typing import Iterable, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table


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
        default="**/bulk_*.csv",
        help="Glob pattern for CSV files (default: **/bulk_*.csv).",
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
        "--mode",
        choices=("ascii", "matplotlib"),
        default="ascii",
        help="Render mode (default: ascii).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render once and exit (no live updates).",
    )
    return parser.parse_args()


def find_csv_files(csv_dir: str, pattern: str) -> list[str]:
    search_pattern = os.path.join(csv_dir, pattern)
    return sorted(glob.glob(search_pattern, recursive=True))


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


def _sample_values(values: list[float], width: int) -> list[float]:
    if width <= 0:
        return []
    if len(values) <= width:
        return values
    step = len(values) / width
    sampled = []
    for i in range(width):
        index = int(i * step)
        sampled.append(values[index])
    return sampled


def _sparkline(values: list[float], width: int, min_value: float, max_value: float) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    span = max(max_value - min_value, 1e-12)
    sampled = _sample_values(values, width)
    chars = []
    for value in sampled:
        normalized = (value - min_value) / span
        idx = min(len(blocks) - 1, max(0, int(normalized * (len(blocks) - 1))))
        chars.append(blocks[idx])
    return "".join(chars)


def _format_range(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return "n/a"
    return f"{min(values):.4f}-{max(values):.4f}"


def render_ascii_table(series_list: list[CsvSeries], x_axis_label: str) -> Table:
    table = Table(title="CSV Logs (train/val loss)", show_lines=True)
    table.add_column("run")
    table.add_column(f"{x_axis_label} range", justify="right")
    table.add_column("train", no_wrap=True)
    table.add_column("val", no_wrap=True)
    table.add_column("train range", justify="right")
    table.add_column("val range", justify="right")

    term_width = get_terminal_size((120, 30)).columns
    spark_width = max(10, min(80, term_width - 60))

    if not series_list:
        table.add_row("no matching files", "-", "-", "-", "-", "-")
        return table

    for series in series_list:
        if not series.x_values:
            table.add_row(series.name, "-", "-", "-", "-", "-")
            continue
        combined_min = min(series.train_values + series.val_values)
        combined_max = max(series.train_values + series.val_values)
        x_range = f"{series.x_values[0]:.0f}-{series.x_values[-1]:.0f}"
        train_spark = _sparkline(series.train_values, spark_width, combined_min, combined_max)
        val_spark = _sparkline(series.val_values, spark_width, combined_min, combined_max)
        table.add_row(
            series.name,
            x_range,
            train_spark,
            val_spark,
            _format_range(series.train_values),
            _format_range(series.val_values),
        )
    return table


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
    console = Console()
    use_matplotlib = args.mode == "matplotlib"
    if use_matplotlib:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, _ = plt.subplots()
        fig.canvas.manager.set_window_title("nanoGPT CSV Log Viewer")

    last_signatures: Optional[dict[str, tuple[float, int]]] = None
    try:
        if use_matplotlib:
            while True:
                paths = find_csv_files(args.csv_dir, args.pattern)
                signatures = get_file_signatures(paths)
                if last_signatures is None or signatures != last_signatures or args.once:
                    series_list = collect_series(paths, x_col_index, args.max_points)
                    plt.cla()
                    if not series_list:
                        plt.title("CSV Logs (no matching files found)")
                        plt.xlabel(args.x_axis)
                        plt.ylabel("loss")
                    else:
                        for series in series_list:
                            if not series.x_values:
                                continue
                            plt.plot(series.x_values, series.train_values, label=f"{series.name} train")
                            plt.plot(series.x_values, series.val_values, label=f"{series.name} val")
                        plt.title("CSV Logs (train/val loss)")
                        plt.xlabel(args.x_axis)
                        plt.ylabel("loss")
                        plt.grid(True, alpha=0.3)
                        plt.legend(fontsize="small")
                        plt.tight_layout()
                    last_signatures = signatures
                    plt.pause(0.1)
                if args.once:
                    break
                time.sleep(args.refresh_seconds)
        else:
            with Live(console=console, refresh_per_second=4, screen=True) as live:
                while True:
                    paths = find_csv_files(args.csv_dir, args.pattern)
                    signatures = get_file_signatures(paths)
                    if last_signatures is None or signatures != last_signatures or args.once:
                        series_list = collect_series(paths, x_col_index, args.max_points)
                        table = render_ascii_table(series_list, args.x_axis)
                        live.update(table)
                        last_signatures = signatures
                    if args.once:
                        break
                    time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
