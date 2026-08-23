#!/usr/bin/env python3
"""Sweep isotropic INT/TQ angular distortion over power-of-two dimensions."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from turboquant_angular_distortion import (
    angle_deg,
    int_quantizer,
    pair_at_angle,
    tq_line_colors,
    tq_quantizer,
)


@dataclass(frozen=True)
class CurveMetric:
    format: str
    bits: int
    dim: int
    angle_deg: float
    mean_distortion_deg: float
    std_distortion_deg: float


@dataclass(frozen=True)
class SummaryMetric:
    format: str
    bits: int
    dim: int
    mean_absolute_distortion_deg: float
    rms_distortion_deg: float
    max_absolute_distortion_deg: float
    signed_bias_deg: float


def power_of_two_dimensions(min_dim: int, max_dim: int) -> list[int]:
    """Return all powers of two in the inclusive requested range."""
    if min_dim < 2 or max_dim < min_dim:
        raise ValueError("Require 2 <= min_dim <= max_dim.")
    dimensions = []
    dim = 2
    while dim <= max_dim:
        if dim >= min_dim:
            dimensions.append(dim)
        dim *= 2
    if not dimensions:
        raise ValueError("The requested range contains no power-of-two dimensions.")
    return dimensions


def collect_metrics(dimensions: list[int], angles: np.ndarray, bits_list: list[int],
                    trials: int, clip_sigma: float,
                    seed: int) -> tuple[list[CurveMetric], list[SummaryMetric]]:
    """Evaluate every format on shared isotropic pairs for fair comparisons."""
    rng = np.random.default_rng(seed)
    curve_rows = []
    summary_rows = []

    for dim in dimensions:
        quantizers = {
            **{f"INT{bits}": int_quantizer(bits, dim, clip_sigma) for bits in bits_list},
            **{f"TQ{bits}": tq_quantizer(bits, dim) for bits in bits_list},
        }
        distortions = {
            name: np.empty((len(angles), trials), dtype=np.float64)
            for name in quantizers
        }

        for angle_index, angle in enumerate(angles):
            for trial in range(trials):
                x, y = pair_at_angle(dim, float(angle), "isotropic", rng)
                for name, quantizer in quantizers.items():
                    distortions[name][angle_index, trial] = (
                        angle_deg(quantizer(x), quantizer(y)) - angle
                    )

        for name, values in distortions.items():
            bits = int(''.join(filter(str.isdigit, name)))
            means = np.mean(values, axis=1)
            stds = np.std(values, axis=1, ddof=1) if trials > 1 else np.zeros(len(angles))
            curve_rows.extend(
                CurveMetric(name, bits, dim, float(angle), float(mean), float(std))
                for angle, mean, std in zip(angles, means, stds)
            )
            summary_rows.append(SummaryMetric(
                name, bits, dim,
                float(np.mean(np.abs(means))),
                float(np.sqrt(np.mean(means ** 2))),
                float(np.max(np.abs(means))),
                float(np.mean(means)),
            ))

    return curve_rows, summary_rows


def write_dataclass_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: getattr(row, field) for field in fields} for row in rows)


def plot_angle_curves(path: Path, curve_rows: list[CurveMetric],
                      dimensions: list[int], bits_list: list[int]) -> None:
    """Write one panel per bit width with dimension-colored INT/TQ curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, math.ceil(len(bits_list) / 2), figsize=(16, 9),
                             constrained_layout=True, squeeze=False)
    dimension_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(dimensions)))
    lookup = {(row.format, row.dim, row.angle_deg): row for row in curve_rows}
    angles = sorted(set(row.angle_deg for row in curve_rows))

    for axis, bits in zip(axes.flat, bits_list):
        for color, dim in zip(dimension_colors, dimensions):
            for prefix, linestyle in (("INT", "-"), ("TQ", ":")):
                values = [lookup[(f"{prefix}{bits}", dim, angle)].mean_distortion_deg
                          for angle in angles]
                axis.plot(angles, values, color=color, linestyle=linestyle,
                          linewidth=1.3)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(f"{bits}-bit formats")
        axis.set_xlabel("Original angle (degrees)")
        axis.set_ylabel("Angular distortion (degrees)")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(bits_list):]:
        axis.set_visible(False)

    handles = [Line2D([], [], color=color, label=f"d={dim}")
               for color, dim in zip(dimension_colors, dimensions)]
    handles.extend([Line2D([], [], color="black", linestyle="-", label="INT"),
                    Line2D([], [], color="black", linestyle=":", label="TQ")])
    fig.legend(handles=handles, loc="outside lower center", ncol=6)
    fig.suptitle("Isotropic angular distortion across dimensions")
    fig.savefig(path)
    plt.close(fig)


def plot_summary(path: Path, rows: list[SummaryMetric], bits_list: list[int]) -> None:
    """Plot dimension scaling of aggregate distortion metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("mean_absolute_distortion_deg", "Mean absolute distortion"),
        ("rms_distortion_deg", "RMS distortion"),
        ("max_absolute_distortion_deg", "Maximum absolute distortion"),
        ("signed_bias_deg", "Signed distortion bias"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    tq_colors = tq_line_colors(len(bits_list))
    lookup = {(row.format, row.dim): row for row in rows}
    dimensions = sorted(set(row.dim for row in rows))

    for axis, (field, title) in zip(axes.flat, panels):
        for color, bits in zip(tq_colors, bits_list):
            for prefix, linestyle, marker in (("INT", "-", "o"), ("TQ", ":", "s")):
                values = [getattr(lookup[(f"{prefix}{bits}", dim)], field)
                          for dim in dimensions]
                axis.plot(dimensions, values, color=color, linestyle=linestyle,
                          marker=marker, markersize=3, label=f"{prefix}{bits}")
        axis.set_xscale("log", base=2)
        axis.set_xticks(dimensions, [str(dim) for dim in dimensions])
        axis.set_title(title)
        axis.set_xlabel("Dimension")
        axis.set_ylabel("Degrees")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Isotropic distortion scaling from d={dimensions[0]} to d={dimensions[-1]}")
    fig.savefig(path)
    plt.close(fig)


def plot_tq_advantage(path: Path, rows: list[SummaryMetric], bits_list: list[int]) -> None:
    """Plot INT minus TQ aggregate error; positive values favor TurboQuant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = {(row.format, row.dim): row for row in rows}
    dimensions = sorted(set(row.dim for row in rows))
    colors = tq_line_colors(len(bits_list))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for color, bits in zip(colors, bits_list):
        advantage = [
            lookup[(f"INT{bits}", dim)].mean_absolute_distortion_deg
            - lookup[(f"TQ{bits}", dim)].mean_absolute_distortion_deg
            for dim in dimensions
        ]
        ax.plot(dimensions, advantage, color=color, marker="o", label=f"{bits}-bit")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dimensions, [str(dim) for dim in dimensions])
    ax.set(title="TurboQuant angular-distortion advantage by dimension",
           xlabel="Dimension",
           ylabel="INT MAE - TQ MAE (degrees; positive favors TQ)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-dim", type=int, default=2)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--bits", type=int, nargs="+", default=list(range(3, 9)))
    parser.add_argument("--angles-start", type=float, default=0.0)
    parser.add_argument("--angles-stop", type=float, default=90.0)
    parser.add_argument("--angles-step", type=float, default=5.0)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/isotropic_dimension_sweep"))
    args = parser.parse_args()
    if args.trials < 1 or args.angles_step <= 0:
        raise SystemExit("--trials and --angles-step must be positive")

    dimensions = power_of_two_dimensions(args.min_dim, args.max_dim)
    angles = np.arange(args.angles_start, args.angles_stop + args.angles_step / 2,
                       args.angles_step)
    curves, summaries = collect_metrics(dimensions, angles, args.bits, args.trials,
                                         args.clip_sigma, args.seed)
    output = args.output_dir
    write_dataclass_csv(output / "isotropic_dimension_curves.csv", curves)
    write_dataclass_csv(output / "isotropic_dimension_summary.csv", summaries)
    plot_angle_curves(output / "isotropic_dimension_curves.pdf", curves,
                      dimensions, args.bits)
    plot_summary(output / "isotropic_dimension_summary.pdf", summaries, args.bits)
    plot_tq_advantage(output / "isotropic_dimension_tq_advantage.pdf", summaries,
                      args.bits)
    print(f"Wrote isotropic dimension sweep outputs to {output}")


if __name__ == "__main__":
    main()
