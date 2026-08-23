#!/usr/bin/env python3
"""Sweep grouped scalar/power-of-two INT quantizers and native block formats."""

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

from low_bit_em_comparison import all_finite_em_values, nearest_values
from turboquant_angular_distortion import angle_deg, pair_at_angle


GROUP_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
METHOD_STYLES = {
    "symmetric_scalar": ("#277da1", "-", "o"),
    "asymmetric_scalar": ("#43aa8b", "--", "s"),
    "symmetric_pow2": ("#f8961e", "-.", "^"),
    "asymmetric_pow2": ("#9b5de5", ":", "D"),
}


@dataclass(frozen=True)
class GroupCurveMetric:
    method: str
    group_size: int
    angle_deg: float
    mean_distortion_deg: float
    sem_distortion_deg: float
    mean_nmse: float
    sem_nmse: float


@dataclass(frozen=True)
class GroupSummaryMetric:
    method: str
    group_size: int
    mean_absolute_distortion_deg: float
    sem_mean_absolute_distortion_deg: float
    mean_nmse: float
    sem_nmse: float


def ceil_power_of_two(value: np.ndarray) -> np.ndarray:
    """Round positive scales upward to powers of two to prevent overflow."""
    tiny = np.finfo(np.float64).tiny
    return np.exp2(np.ceil(np.log2(np.maximum(value, tiny))))


def grouped_int_quantize(values: np.ndarray, group_size: int, bits: int = 4,
                         asymmetric: bool = False,
                         power_of_two_scale: bool = False) -> np.ndarray:
    """Quantize contiguous groups with exact or power-of-two constrained scales."""
    if values.size % group_size:
        raise ValueError("Vector length must be divisible by group size.")
    groups = np.asarray(values, dtype=np.float64).reshape(-1, group_size)
    if asymmetric:
        qmin, qmax = 0, 2 ** bits - 1
        minimum = np.min(groups, axis=1, keepdims=True)
        maximum = np.max(groups, axis=1, keepdims=True)
        scale = (maximum - minimum) / (qmax - qmin)
        scale = np.maximum(scale, np.finfo(float).tiny)
        if power_of_two_scale:
            scale = ceil_power_of_two(scale)
        zero_point = np.clip(np.rint(qmin - minimum / scale), qmin, qmax)
        codes = np.clip(np.rint(groups / scale + zero_point), qmin, qmax)
        output = (codes - zero_point) * scale
    else:
        qmax = 2 ** (bits - 1) - 1
        scale = np.max(np.abs(groups), axis=1, keepdims=True) / qmax
        scale = np.maximum(scale, np.finfo(float).tiny)
        if power_of_two_scale:
            scale = ceil_power_of_two(scale)
        codes = np.clip(np.rint(groups / scale), -qmax, qmax)
        output = codes * scale
    return output.reshape(values.shape)


def quantize_e4m3_scale(values: np.ndarray) -> np.ndarray:
    """Quantize positive block scales as E4M3FN (maximum finite 448)."""
    codebook = all_finite_em_values(4, 3)
    positive = codebook[(codebook >= 0) & (codebook <= 448.0)]
    return nearest_values(values, positive)


def nvfp4_quantize(values: np.ndarray) -> np.ndarray:
    """Emulate native NVFP4: E2M1 data, 16-value E4M3 scales, FP32 tensor scale."""
    groups = np.asarray(values, dtype=np.float64).reshape(-1, 16)
    e2m1 = all_finite_em_values(2, 1)
    local_scale = np.max(np.abs(groups), axis=1, keepdims=True) / 6.0
    tensor_scale = max(float(np.max(local_scale)) / 448.0, np.finfo(float).tiny)
    stored_scale = quantize_e4m3_scale(local_scale / tensor_scale) * tensor_scale
    stored_scale = np.maximum(stored_scale, np.finfo(float).tiny)
    output = nearest_values(groups / stored_scale, e2m1) * stored_scale
    return output.reshape(values.shape)


def mxint_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    """Emulate MXINT with 32-value blocks and an E8M0 power-of-two scale."""
    return grouped_int_quantize(values, 32, bits=bits, asymmetric=False,
                                power_of_two_scale=True)


def method_quantizers(group_size: int) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "symmetric_scalar": lambda x: grouped_int_quantize(x, group_size),
        "asymmetric_scalar": lambda x: grouped_int_quantize(x, group_size, asymmetric=True),
        "symmetric_pow2": lambda x: grouped_int_quantize(
            x, group_size, power_of_two_scale=True),
        "asymmetric_pow2": lambda x: grouped_int_quantize(
            x, group_size, asymmetric=True, power_of_two_scale=True),
    }


def collect_metrics(dimension: int, group_sizes: list[int], angles: np.ndarray,
                    trials: int, seed: int
                    ) -> tuple[list[GroupCurveMetric], list[GroupSummaryMetric]]:
    """Evaluate grouped INT4 sweeps plus native NVFP4/MXINT references."""
    rng = np.random.default_rng(seed)
    quantizers = {
        **{f"{method}:g{group}": quantizer
           for group in group_sizes for method, quantizer in method_quantizers(group).items()},
        "NVFP4:g16": nvfp4_quantize,
        "MXINT8:g32": lambda x: mxint_quantize(x, 8),
    }
    distortion = {name: np.empty((len(angles), trials)) for name in quantizers}
    nmse = {name: np.empty((len(angles), trials)) for name in quantizers}
    for angle_index, angle in enumerate(angles):
        for trial in range(trials):
            x, y = pair_at_angle(dimension, float(angle), "isotropic", rng)
            for name, quantizer in quantizers.items():
                qx, qy = quantizer(x), quantizer(y)
                distortion[name][angle_index, trial] = angle_deg(qx, qy) - angle
                nmse[name][angle_index, trial] = 0.5 * (
                    np.sum((qx - x) ** 2) + np.sum((qy - y) ** 2)
                )

    curves, summaries = [], []
    for name in quantizers:
        method, group_text = name.split(":g")
        group_size = int(group_text)
        dist_std = np.std(distortion[name], axis=1, ddof=1) if trials > 1 else np.zeros(len(angles))
        nmse_std = np.std(nmse[name], axis=1, ddof=1) if trials > 1 else np.zeros(len(angles))
        curves.extend(
            GroupCurveMetric(method, group_size, float(angle),
                             float(np.mean(distortion[name][index])),
                             float(dist_std[index] / math.sqrt(trials)),
                             float(np.mean(nmse[name][index])),
                             float(nmse_std[index] / math.sqrt(trials)))
            for index, angle in enumerate(angles)
        )
        trial_mae = np.mean(np.abs(distortion[name]), axis=0)
        trial_nmse = np.mean(nmse[name], axis=0)
        summaries.append(GroupSummaryMetric(
            method, group_size, float(np.mean(trial_mae)),
            float(np.std(trial_mae, ddof=1) / math.sqrt(trials)) if trials > 1 else 0.0,
            float(np.mean(trial_nmse)),
            float(np.std(trial_nmse, ddof=1) / math.sqrt(trials)) if trials > 1 else 0.0,
        ))
    return curves, summaries


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot_group_summary(path: Path, summaries: list[GroupSummaryMetric]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    native_styles = {"NVFP4": ("#e63946", "*"), "MXINT8": ("#1982c4", "X")}
    for axis, field, error_field, title in (
        (axes[0], "mean_absolute_distortion_deg", "sem_mean_absolute_distortion_deg",
         "Mean absolute angular distortion"),
        (axes[1], "mean_nmse", "sem_nmse", "Normalized reconstruction MSE"),
    ):
        for method, (color, linestyle, marker) in METHOD_STYLES.items():
            rows = sorted((row for row in summaries if row.method == method),
                          key=lambda row: row.group_size)
            axis.errorbar([row.group_size for row in rows], [getattr(row, field) for row in rows],
                          yerr=[getattr(row, error_field) for row in rows], label=method,
                          color=color, linestyle=linestyle, marker=marker, capsize=2)
        for method, (color, marker) in native_styles.items():
            row = next(row for row in summaries if row.method == method)
            axis.scatter(row.group_size, getattr(row, field), color=color, marker=marker,
                         s=100, label=f"{method} native g={row.group_size}", zorder=5)
        axis.set_xscale("log", base=2)
        axis.set_xticks(GROUP_SIZES, [str(group) for group in GROUP_SIZES])
        axis.set(title=title, xlabel="Group size")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Grouped quantization sweep (d=2048, INT4 sweep)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_angle_curves(path: Path, curves: list[GroupCurveMetric]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    group_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(GROUP_SIZES)))
    for axis, (method, (_, linestyle, _)) in zip(axes.flat, METHOD_STYLES.items()):
        for color, group in zip(group_colors, GROUP_SIZES):
            rows = [row for row in curves if row.method == method and row.group_size == group]
            angles = np.asarray([row.angle_deg for row in rows])
            means = np.asarray([row.mean_distortion_deg for row in rows])
            errors = np.asarray([row.sem_distortion_deg for row in rows])
            axis.plot(angles, means, color=color, linestyle=linestyle, label=f"g={group}")
            axis.fill_between(angles, means - errors, means + errors, color=color, alpha=0.06)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set(title=method.replace("_", " ").title(), xlabel="Original angle (degrees)",
                 ylabel="Angular distortion (degrees)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Symmetric/asymmetric scalar and power-of-two grouped INT4")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--angles-step", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/grouped_quantization_sweep"))
    args = parser.parse_args()
    if args.dimension != 2048:
        raise SystemExit("This comparison currently targets --dimension 2048.")
    angles = np.arange(0.0, 90.0 + args.angles_step / 2, args.angles_step)
    curves, summaries = collect_metrics(args.dimension, GROUP_SIZES, angles,
                                         args.trials, args.seed)
    output = args.output_dir
    write_csv(output / "grouped_quantization_curves.csv", curves)
    write_csv(output / "grouped_quantization_summary.csv", summaries)
    plot_angle_curves(output / "grouped_quantization_angle_curves.pdf", curves)
    plot_group_summary(output / "grouped_quantization_group_sweep.pdf", summaries)
    print(f"Wrote grouped quantization sweep outputs to {output}")


if __name__ == "__main__":
    main()
