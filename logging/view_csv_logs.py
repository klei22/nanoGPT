#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass
from shutil import get_terminal_size
from typing import Optional, TypeVar

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


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
        "--layout",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="ASCII layout for legend placement (default: vertical).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=20,
        help="ASCII chart height in rows (default: 20).",
    )
    parser.add_argument(
        "--max-x",
        type=float,
        default=3000.0,
        help="Maximum X-axis value to scale the plot (default: 3000).",
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


SampledType = TypeVar("SampledType")


def _sample_values(values: list[SampledType], width: int) -> list[SampledType]:
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


def _build_plot_lines(
    width: int,
    height: int,
    series_list: list[CsvSeries],
    min_value: float,
    max_value: float,
    max_x: float,
    train_styles: list[str],
    val_styles: list[str],
) -> list[Text]:
    grid: list[list[Optional[tuple[str, str]]]] = [
        [None for _ in range(width)] for _ in range(height)
    ]
    span = max(max_value - min_value, 1e-12)

    for idx, series in enumerate(series_list):
        if not series.x_values:
            continue
        train_style = train_styles[idx % len(train_styles)]
        val_style = val_styles[idx % len(val_styles)]
        train_points = list(zip(series.x_values, series.train_values))
        val_points = list(zip(series.x_values, series.val_values))

        def draw_series(points: list[tuple[float, float]], style: str, char: str) -> None:
            if not points:
                return
            sampled_points = _sample_values(points, width)
            last_x = None
            last_y = None
            for x_value, y_value in sampled_points:
                clamped_x = max(0.0, min(max_x, x_value))
                x_pos = int((clamped_x / max_x) * (width - 1)) if max_x > 0 else 0
                normalized = (y_value - min_value) / span
                y_pos = height - 1 - int(normalized * (height - 1))
                if last_x is not None and last_y is not None:
                    dx = x_pos - last_x
                    dy = y_pos - last_y
                    steps = max(abs(dx), abs(dy), 1)
                    for step in range(steps + 1):
                        interp_x = int(last_x + dx * (step / steps))
                        interp_y = int(last_y + dy * (step / steps))
                        grid[interp_y][interp_x] = (char, style)
                else:
                    grid[y_pos][x_pos] = (char, style)
                last_x = x_pos
                last_y = y_pos

        draw_series(train_points, train_style, "─")
        draw_series(val_points, val_style, "═")

    lines: list[Text] = []
    for row in grid:
        line = Text()
        for cell in row:
            if cell is None:
                line.append(" ")
            else:
                char, style = cell
                line.append(char, style=style)
        lines.append(line)
    return lines


def render_ascii_plot(
    series_list: list[CsvSeries],
    x_axis_label: str,
    layout: str,
    height: int,
    max_x: float,
) -> Group:
    term_width = get_terminal_size((120, 30)).columns
    label_width = 10
    plot_width = max(20, min(120, term_width - label_width - 6))

    if not series_list:
        empty_text = Text("No matching files found.", style="bold yellow")
        return Group(Panel(empty_text, title="CSV Logs (train/val loss)"))

    combined_values = [
        value
        for series in series_list
        for value in (series.train_values + series.val_values)
    ]
    if not combined_values:
        empty_text = Text("No data points yet.", style="bold yellow")
        return Group(Panel(empty_text, title="CSV Logs (train/val loss)"))
    min_value = min(combined_values)
    max_value = max(combined_values)

    train_styles = ["red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    val_styles = [
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_white",
    ]

    plot_lines = _build_plot_lines(
        plot_width,
        max(5, height),
        series_list,
        min_value,
        max_value,
        max_x,
        train_styles,
        val_styles,
    )
    tick_count = 5
    tick_values = [
        min_value + i * (max_value - min_value) / (tick_count - 1)
        for i in range(tick_count)
    ]
    tick_rows = {
        max(
            0,
            min(
                len(plot_lines) - 1,
                len(plot_lines) - 1 - int(i * (len(plot_lines) - 1) / (tick_count - 1)),
            ),
        )
        for i in range(tick_count)
    }
    labeled_lines: list[Text] = []
    for row_idx, line in enumerate(plot_lines):
        label = ""
        if row_idx in tick_rows:
            tick_index = tick_count - 1 - int(row_idx * (tick_count - 1) / (len(plot_lines) - 1))
            label = f"{tick_values[tick_index]:>7.3f}"
        prefix = Text(f"{label:>{label_width}} │", style="dim")
        combined = Text()
        combined.append_text(prefix)
        combined.append_text(line)
        labeled_lines.append(combined)

    axis_line = Text(f"{' ' * label_width} └" + "─" * plot_width, style="dim")
    tick_positions = [0, plot_width // 2, plot_width - 1]
    tick_labels = [f"0", f"{max_x / 2:.0f}", f"{max_x:.0f}"]
    axis_labels = Text(" " * (label_width + 2))
    last_pos = 0
    for pos, label in zip(tick_positions, tick_labels):
        spacing = max(0, pos - last_pos)
        axis_labels.append(" " * spacing)
        axis_labels.append(label)
        last_pos = pos + len(label)
    axis_info = Text(
        f"{x_axis_label} (max {max_x:.0f}) | loss range {min_value:.4f}-{max_value:.4f}",
        style="dim",
    )

    legend_lines: list[Text] = []
    for idx, series in enumerate(series_list):
        train_style = train_styles[idx % len(train_styles)]
        val_style = val_styles[idx % len(val_styles)]
        best_val = None
        best_iter = None
        if series.val_values and series.x_values:
            best_val = min(series.val_values)
            best_index = series.val_values.index(best_val)
            if best_index < len(series.x_values):
                best_iter = series.x_values[best_index]
        legend = Text()
        legend.append("─ train ", style=train_style)
        legend.append("═ val ", style=val_style)
        legend.append(series.name, style="bold")
        if best_val is not None and best_iter is not None:
            legend.append(f" | best val {best_val:.4f} @ {best_iter:.0f}")
        legend_lines.append(legend)

    plot_panel = Panel(
        Group(*labeled_lines, axis_line, axis_labels, axis_info),
        title="CSV Logs (train/val loss)",
        padding=(0, 1),
    )
    legend_panel = Panel(
        Group(*legend_lines) if legend_lines else Text("No runs"),
        title="Legend",
    )

    if layout == "horizontal":
        return Group(Columns([plot_panel, legend_panel], expand=True))
    return Group(plot_panel, legend_panel)


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
                        plot = render_ascii_plot(
                            series_list,
                            args.x_axis,
                            args.layout,
                            args.height,
                            args.max_x,
                        )
                        live.update(plot)
                        last_signatures = signatures
                    if args.once:
                        break
                    time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
