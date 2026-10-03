#!/usr/bin/env python3
"""Measure angular-space evenness for scalar codebooks above three dimensions."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betainc

from angle_space_evenness import format_values


@dataclass(frozen=True)
class HighDimEvennessMetric:
    format: str
    dimension: int
    samples: int
    mean_cosine: float
    cosine_std: float
    uniform_cosine_std: float
    cosine_std_error: float
    cosine_ks_discrepancy: float
    resultant_norm: float
    second_moment_anisotropy: float


def sphere_cosine_cdf(values: np.ndarray, dimension: int) -> np.ndarray:
    """Exact cosine CDF for two independent uniform points on S^(d-1)."""
    shape = 0.5 * (dimension - 1)
    return betainc(shape, shape, np.clip((values + 1.0) / 2.0, 0.0, 1.0))


def evaluate_format(name: str, dimension: int, samples: int, batch_size: int,
                    seed: int) -> HighDimEvennessMetric:
    """Stream random code vectors and compare pair cosines to uniform-sphere theory."""
    rng = np.random.default_rng(seed)
    values = format_values(name)
    cosines = []
    component_sum = 0.0
    component_count = 0
    off_diagonal_sum = 0.0
    completed = 0
    while completed < samples:
        count = min(batch_size, samples - completed)
        left = rng.choice(values, size=(count, dimension))
        right = rng.choice(values, size=(count, dimension))
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        cosines.append(np.sum(left * right, axis=1))
        component_sum += float(np.sum(left) + np.sum(right))
        component_count += 2 * count * dimension
        row_sums = np.sum(left, axis=1)
        off_diagonal_sum += float(np.sum(row_sums ** 2 - 1.0))
        completed += count

    cosine = np.sort(np.concatenate(cosines))
    empirical = np.arange(1, samples + 1, dtype=np.float64) / samples
    theoretical = sphere_cosine_cdf(cosine, dimension)
    ks = float(np.max(np.maximum(np.abs(empirical - theoretical),
                                 np.abs((empirical - 1.0 / samples) - theoretical))))
    mean_component = component_sum / component_count
    resultant = math.sqrt(dimension) * abs(mean_component)
    off_diagonal = off_diagonal_sum / (samples * dimension * (dimension - 1))
    parallel_eigenvalue = 1.0 / dimension + (dimension - 1) * off_diagonal
    perpendicular_eigenvalue = 1.0 / dimension - off_diagonal
    anisotropy = math.sqrt(
        (dimension * parallel_eigenvalue - 1.0) ** 2
        + (dimension - 1) * (dimension * perpendicular_eigenvalue - 1.0) ** 2
    )
    observed_std = float(np.std(cosine, ddof=1))
    expected_std = 1.0 / math.sqrt(dimension)
    return HighDimEvennessMetric(
        name.upper(), dimension, samples, float(np.mean(cosine)), observed_std,
        expected_std, observed_std - expected_std, ks, resultant, anisotropy,
    )


def write_outputs(output_dir: Path, dimension: int,
                  rows: list[HighDimEvennessMetric]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (
        -int(''.join(filter(str.isdigit, row.format))),
        0 if row.format.startswith("INT") else 1,
    ))
    csv_path = output_dir / f"angle_space_evenness_d{dimension}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    panels = [
        ("cosine_std_error", "Cosine std error", "lower |value| is better"),
        ("cosine_ks_discrepancy", "Cosine CDF discrepancy", "lower is better"),
        ("resultant_norm", "Resultant/dipole norm", "lower is better"),
        ("second_moment_anisotropy", "Second-moment anisotropy", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    names = [row.format for row in rows]
    colors = ["#277da1" if name.startswith("INT") else "#f94144" for name in names]
    for axis, (field, title, subtitle) in zip(axes.flat, panels):
        axis.bar(names, [getattr(row, field) for row in rows], color=colors)
        axis.set_title(f"{title}\n({subtitle})")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"High-dimensional angular-space evenness (d={dimension})")
    fig.savefig(output_dir / f"angle_space_evenness_d{dimension}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--formats", nargs="+",
                        default=[*(f"int{i}" for i in range(3, 9)),
                                 *(f"tq{i}" for i in range(3, 9))])
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/high_dim_angle_space_evenness"))
    args = parser.parse_args()
    for dimension in args.dimensions:
        if dimension < 2:
            raise SystemExit("All dimensions must be at least 2")
        rows = [evaluate_format(name, dimension, args.samples, args.batch_size,
                                args.seed + index)
                for index, name in enumerate(args.formats)]
        write_outputs(args.output_dir, dimension, rows)
    print(f"Wrote high-dimensional evenness outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
