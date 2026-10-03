#!/usr/bin/env python3
"""Measure how evenly scalar number formats cover angular space on S2."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from turboquant_angular_distortion import gaussian_lloyd_max_codebook


@dataclass(frozen=True)
class EvennessMetrics:
    format: str
    weighting: str
    samples: int
    occupied_pixels: int
    coverage: float
    normalized_entropy: float
    effective_coverage: float
    js_divergence: float
    count_cv: float
    cap_discrepancy: float
    dipole_norm: float
    quadrupole_error: float


def format_values(name: str) -> np.ndarray:
    """Return the scalar reconstruction values for an INT or TQ format."""
    lowered = name.lower()
    if lowered.startswith("int"):
        bits = int(lowered[3:])
        if bits < 2 or bits > 16:
            raise ValueError("INT bits must be between 2 and 16")
        return np.arange(-(2 ** (bits - 1)), 2 ** (bits - 1), dtype=np.float64)
    if lowered.startswith("tq"):
        return gaussian_lloyd_max_codebook(int(lowered[2:]))
    raise ValueError(f"Unsupported format {name!r}; use INT<bits> or TQ<bits>.")


def sample_directions(values: np.ndarray, count: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample scalar-code triples and project nonzero triples to S2."""
    directions = []
    remaining = count
    while remaining:
        triples = rng.choice(values, size=(max(remaining, 1024), 3))
        norms = np.linalg.norm(triples, axis=1)
        valid = triples[norms > 0]
        valid /= np.linalg.norm(valid, axis=1, keepdims=True)
        take = min(remaining, valid.shape[0])
        directions.append(valid[:take])
        remaining -= take
    return np.concatenate(directions)


def histogram_metrics(histogram: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return coverage, entropy, effective coverage, JS divergence, and CV."""
    counts = np.asarray(histogram, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("Histogram must contain at least one observation.")
    probabilities = counts / total
    npix = counts.size
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    normalized_entropy = entropy / math.log(npix)
    effective_coverage = math.exp(entropy) / npix
    uniform = np.full(npix, 1.0 / npix)
    mixture = 0.5 * (probabilities + uniform)
    js = 0.5 * float(np.sum(probabilities[positive] *
                            np.log(probabilities[positive] / mixture[positive])))
    js += 0.5 * float(np.sum(uniform * np.log(uniform / mixture)))
    coverage = float(np.count_nonzero(counts)) / npix
    count_cv = float(np.std(counts) / np.mean(counts))
    return coverage, normalized_entropy, effective_coverage, js, count_cv


def spherical_cap_discrepancy(directions: np.ndarray, cap_count: int,
                              rng: np.random.Generator) -> float:
    """Estimate max |empirical cap mass - uniform cap area| over random caps."""
    centers = rng.normal(size=(cap_count, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Uniform cos(radius) makes cap areas uniformly distributed in [0, 1].
    thresholds = rng.uniform(-1.0, 1.0, size=cap_count)
    maximum = 0.0
    for start in range(0, cap_count, 32):
        center_block = centers[start:start + 32]
        threshold_block = thresholds[start:start + 32]
        observed = np.mean(directions @ center_block.T >= threshold_block, axis=0)
        expected = 0.5 * (1.0 - threshold_block)
        maximum = max(maximum, float(np.max(np.abs(observed - expected))))
    return maximum


def moment_metrics(directions: np.ndarray) -> tuple[float, float]:
    """Return first-moment (dipole) and second-moment (quadrupole) anisotropy."""
    dipole = float(np.linalg.norm(np.mean(directions, axis=0)))
    second_moment = directions.T @ directions / directions.shape[0]
    quadrupole = float(np.linalg.norm(3.0 * second_moment - np.eye(3), ord="fro"))
    return dipole, quadrupole


def metrics_for_directions(name: str, weighting: str, directions: np.ndarray,
                           nside: int, cap_count: int,
                           rng: np.random.Generator) -> EvennessMetrics:
    """Calculate the metric suite for one weighted directional sample."""
    pixels = hp.vec2pix(nside, directions[:, 0], directions[:, 1], directions[:, 2])
    histogram = np.bincount(pixels, minlength=hp.nside2npix(nside))
    coverage, entropy, effective, js, cv = histogram_metrics(histogram)
    cap_discrepancy = spherical_cap_discrepancy(directions, cap_count, rng)
    dipole, quadrupole = moment_metrics(directions)
    return EvennessMetrics(name.upper(), weighting, directions.shape[0],
                           int(np.count_nonzero(histogram)), coverage, entropy,
                           effective, js, cv, cap_discrepancy, dipole, quadrupole)


def evaluate_format(name: str, samples: int, nside: int, cap_count: int,
                    seed: int, weighting: str = "both") -> list[EvennessMetrics]:
    rng = np.random.default_rng(seed)
    directions = sample_directions(format_values(name), samples, rng)
    rows = []
    if weighting in {"codes", "both"}:
        rows.append(metrics_for_directions(name, "codes", directions, nside,
                                           cap_count, rng))
    if weighting in {"unique", "both"}:
        unique = np.unique(np.round(directions, decimals=12), axis=0)
        rows.append(metrics_for_directions(name, "unique", unique, nside,
                                           cap_count, rng))
    return rows


def write_csv(path: Path, rows: list[EvennessMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def ordered_formats(rows: list[EvennessMetrics]) -> list[str]:
    """Order formats by descending bits, with INT immediately before TQ."""
    formats = set(row.format for row in rows)

    def sort_key(name: str) -> tuple[int, int, str]:
        prefix = "INT" if name.startswith("INT") else "TQ"
        bit_text = name.removeprefix(prefix)
        bits = int(bit_text) if bit_text.isdigit() else -1
        return -bits, 0 if prefix == "INT" else 1, name

    return sorted(formats, key=sort_key)


def plot_metrics(path: Path, rows: list[EvennessMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formats = ordered_formats(rows)
    weightings = list(dict.fromkeys(row.weighting for row in rows))
    row_lookup = {(row.format, row.weighting): row for row in rows}
    x_positions = np.arange(len(formats), dtype=np.float64)
    bar_width = 0.8 / len(weightings)
    family_colors = {"INT": "#277da1", "TQ": "#f94144"}
    weighting_hatches = {"codes": "", "unique": "//"}
    panels = [
        ("coverage", "HEALPix coverage", True),
        ("effective_coverage", "Entropy-effective coverage", True),
        ("js_divergence", "Jensen-Shannon divergence", False),
        ("cap_discrepancy", "Spherical-cap discrepancy", False),
        ("count_cv", "HEALPix count CV", False),
        ("quadrupole_error", "Second-moment anisotropy", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, (field, title, higher_is_better) in zip(axes.flat, panels):
        for index, weighting in enumerate(weightings):
            offset = (index - (len(weightings) - 1) / 2.0) * bar_width
            values = [getattr(row_lookup[(name, weighting)], field)
                      if (name, weighting) in row_lookup else np.nan
                      for name in formats]
            colors = [family_colors["INT" if name.startswith("INT") else "TQ"]
                      for name in formats]
            axis.bar(x_positions + offset, values, width=bar_width,
                     color=colors, hatch=weighting_hatches.get(weighting, ""),
                     edgecolor="white", linewidth=0.6)
        axis.set_title(f"{title}\n({'higher' if higher_is_better else 'lower'} is better)")
        axis.set_xticks(x_positions, formats, rotation=35, ha="right")
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        for position in range(1, len(formats)):
            previous_bits = ''.join(filter(str.isdigit, formats[position - 1]))
            current_bits = ''.join(filter(str.isdigit, formats[position]))
            if previous_bits != current_bits:
                axis.axvline(position - 0.5, color="gray", alpha=0.2, linewidth=0.8)
    legend_handles = [
        Patch(facecolor=family_colors["INT"], label="Integer"),
        Patch(facecolor=family_colors["TQ"], label="TurboQuant"),
        *(Patch(facecolor="white", edgecolor="gray",
                hatch=weighting_hatches.get(weighting, ""), label=weighting)
          for weighting in weightings),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4)
    fig.suptitle("Angular-space evenness and dispersion")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+",
                        default=[*(f"int{i}" for i in range(3, 9)),
                                 *(f"tq{i}" for i in range(3, 9))])
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--caps", type=int, default=512)
    parser.add_argument("--weighting", choices=["codes", "unique", "both"],
                        default="both")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--csv", type=Path, default=Path("outputs/angle_space_evenness.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/angle_space_evenness.pdf"))
    args = parser.parse_args()
    if args.samples < 1 or args.caps < 1:
        raise SystemExit("--samples and --caps must be positive")
    if not hp.isnsideok(args.nside):
        raise SystemExit("--nside must be a valid HEALPix NSIDE")

    rows = [row for index, name in enumerate(args.formats)
            for row in evaluate_format(name, args.samples, args.nside, args.caps,
                                       args.seed + index, args.weighting)]
    write_csv(args.csv, rows)
    plot_metrics(args.output, rows)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
