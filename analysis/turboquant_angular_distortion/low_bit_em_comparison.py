#!/usr/bin/env python3
"""Compare 3/4-bit INT, TurboQuant, and all-finite E/M float formats."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from turboquant_angular_distortion import angle_deg, int_quantizer, pair_at_angle, tq_quantizer


LEGAL_EM_FORMATS = {
    3: [(1, 1), (2, 0)],
    4: [(1, 2), (2, 1), (3, 0)],
}

FORMAT_STYLES = {
    "INT": {"color": "#277da1", "linestyle": "-", "marker": "o"},
    "TQ": {"color": "#f94144", "linestyle": ":", "marker": "s"},
    "E1": {"color": "#7b2cbf", "linestyle": "--", "marker": "^"},
    "E2": {"color": "#f8961e", "linestyle": "-.", "marker": "D"},
    "E3": {"color": "#43aa8b", "linestyle": (0, (5, 1, 1, 1)), "marker": "v"},
}


@dataclass(frozen=True)
class LowBitCurveMetric:
    format: str
    total_bits: int
    dimension: int
    angle_deg: float
    mean_distortion_deg: float
    std_distortion_deg: float
    sem_distortion_deg: float


@dataclass(frozen=True)
class LowBitSummaryMetric:
    format: str
    total_bits: int
    dimension: int
    mean_absolute_distortion_deg: float
    sem_mean_absolute_distortion_deg: float


def all_finite_em_values(exp_bits: int, mant_bits: int) -> np.ndarray:
    """Enumerate a sign/E/M format where every exponent code is finite."""
    if exp_bits < 1 or mant_bits < 0:
        raise ValueError("E must be >= 1 and M must be >= 0.")
    bias = 2 ** (exp_bits - 1) - 1
    values = set()
    for sign in (-1.0, 1.0):
        for exponent in range(2 ** exp_bits):
            for mantissa in range(2 ** mant_bits):
                if exponent == 0:
                    magnitude = (2.0 ** (1 - bias)) * mantissa / (2 ** mant_bits)
                else:
                    magnitude = (2.0 ** (exponent - bias)) * (
                        1.0 + mantissa / (2 ** mant_bits)
                    )
                values.add(sign * magnitude)
    return np.asarray(sorted(values), dtype=np.float64)


def nearest_values(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    boundaries = 0.5 * (codebook[:-1] + codebook[1:])
    return codebook[np.searchsorted(boundaries, values)]


def em_quantizer(exp_bits: int, mant_bits: int,
                 dimension: int) -> Callable[[np.ndarray], np.ndarray]:
    """Quantize normalized coordinates after standard-normal scaling."""
    codebook = all_finite_em_values(exp_bits, mant_bits)
    scale = math.sqrt(dimension)
    return lambda values: nearest_values(values * scale, codebook)


def format_quantizers(total_bits: int, dimension: int,
                      clip_sigma: float) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    quantizers = {
        f"INT{total_bits}": int_quantizer(total_bits, dimension, clip_sigma),
        f"TQ{total_bits}": tq_quantizer(total_bits, dimension),
    }
    for exp_bits, mant_bits in LEGAL_EM_FORMATS[total_bits]:
        quantizers[f"E{exp_bits}M{mant_bits}"] = em_quantizer(
            exp_bits, mant_bits, dimension)
    return quantizers


def collect_metrics(dimensions: list[int], angles: np.ndarray, trials: int,
                    clip_sigma: float, seed: int
                    ) -> tuple[list[LowBitCurveMetric], list[LowBitSummaryMetric]]:
    rng = np.random.default_rng(seed)
    curves = []
    summaries = []
    for dimension in dimensions:
        for total_bits in (3, 4):
            quantizers = format_quantizers(total_bits, dimension, clip_sigma)
            values = {name: np.empty((len(angles), trials)) for name in quantizers}
            for angle_index, angle in enumerate(angles):
                for trial in range(trials):
                    x, y = pair_at_angle(dimension, float(angle), "isotropic", rng)
                    for name, quantizer in quantizers.items():
                        values[name][angle_index, trial] = (
                            angle_deg(quantizer(x), quantizer(y)) - angle
                        )
            for name, distortion in values.items():
                means = np.mean(distortion, axis=1)
                stds = (np.std(distortion, axis=1, ddof=1)
                        if trials > 1 else np.zeros(len(angles)))
                curves.extend(
                    LowBitCurveMetric(name, total_bits, dimension, float(angle),
                                      float(mean), float(std),
                                      float(std / math.sqrt(trials)))
                    for angle, mean, std in zip(angles, means, stds)
                )
                trial_mae = np.mean(np.abs(distortion), axis=0)
                summaries.append(LowBitSummaryMetric(
                    name, total_bits, dimension, float(np.mean(trial_mae)),
                    float(np.std(trial_mae, ddof=1) / math.sqrt(trials))
                    if trials > 1 else 0.0,
                ))
    return curves, summaries


def style_for_format(name: str) -> dict[str, object]:
    if name.startswith("INT"):
        return FORMAT_STYLES["INT"]
    if name.startswith("TQ"):
        return FORMAT_STYLES["TQ"]
    return FORMAT_STYLES[name.split("M", 1)[0]]


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot_angle_comparison(path: Path, curves: list[LowBitCurveMetric],
                          dimension: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    selected = [row for row in curves if row.dimension == dimension]
    for axis, total_bits in zip(axes, (3, 4)):
        bit_rows = [row for row in selected if row.total_bits == total_bits]
        formats = list(dict.fromkeys(row.format for row in bit_rows))
        for name in formats:
            rows = [row for row in bit_rows if row.format == name]
            style = style_for_format(name)
            angles = np.asarray([row.angle_deg for row in rows])
            means = np.asarray([row.mean_distortion_deg for row in rows])
            errors = np.asarray([row.sem_distortion_deg for row in rows])
            axis.plot(angles, means, label=name, linewidth=2, **style)
            axis.fill_between(angles, means - errors, means + errors,
                              color=style["color"], alpha=0.1)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set(title=f"{total_bits}-bit formats", xlabel="Original angle (degrees)",
                 ylabel="Angular distortion (degrees)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"INT vs TQ vs all-finite E/M formats (isotropic, d={dimension})")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_dimension_comparison(path: Path, summaries: list[LowBitSummaryMetric]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, total_bits in zip(axes, (3, 4)):
        bit_rows = [row for row in summaries if row.total_bits == total_bits]
        formats = list(dict.fromkeys(row.format for row in bit_rows))
        for name in formats:
            rows = [row for row in bit_rows if row.format == name]
            style = style_for_format(name)
            axis.errorbar([row.dimension for row in rows],
                          [row.mean_absolute_distortion_deg for row in rows],
                          yerr=[row.sem_mean_absolute_distortion_deg for row in rows],
                          label=name, linewidth=2, capsize=2, **style)
        dimensions = sorted(set(row.dimension for row in bit_rows))
        axis.set_xscale("log", base=2)
        axis.set_xticks(dimensions, [str(dim) for dim in dimensions])
        axis.set(title=f"{total_bits}-bit formats", xlabel="Dimension",
                 ylabel="Mean absolute distortion (degrees)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Low-bit isotropic distortion scaling")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--curve-dimension", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--angles-step", type=float, default=5.0)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/low_bit_em_comparison"))
    args = parser.parse_args()
    if args.curve_dimension not in args.dimensions:
        raise SystemExit("--curve-dimension must be included in --dimensions")
    angles = np.arange(0.0, 90.0 + args.angles_step / 2, args.angles_step)
    curves, summaries = collect_metrics(args.dimensions, angles, args.trials,
                                         args.clip_sigma, args.seed)
    output = args.output_dir
    write_csv(output / "low_bit_em_curves.csv", curves)
    write_csv(output / "low_bit_em_summary.csv", summaries)
    plot_angle_comparison(output / "low_bit_em_angular_distortion.pdf", curves,
                          args.curve_dimension)
    plot_dimension_comparison(output / "low_bit_em_dimension_scaling.pdf", summaries)
    print(f"Wrote low-bit E/M comparison outputs to {output}")


if __name__ == "__main__":
    main()
