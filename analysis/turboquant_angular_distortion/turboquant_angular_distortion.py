#!/usr/bin/env python3
"""Compare TurboQuant codebooks and INT3--INT8 angular distortion.

TurboQuant is modeled as a randomized Hadamard transform followed by an
MSE-optimal Gaussian Lloyd-Max scalar codebook.  The same codebook is also run
without the transform to isolate the transform's contribution.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm


def gaussian_lloyd_max_codebook(bits: int, max_iter: int = 10_000,
                                tolerance: float = 1e-12) -> np.ndarray:
    """Return the 2**bits MSE-optimal centroids for N(0, 1)."""
    if bits < 1 or bits > 12:
        raise ValueError("TurboQuant bits must be between 1 and 12.")

    levels = 2 ** bits
    probabilities = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    centroids = norm.ppf(probabilities)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    for _ in range(max_iter):
        boundaries = np.concatenate(([-np.inf],
                                     0.5 * (centroids[:-1] + centroids[1:]),
                                     [np.inf]))
        density = np.exp(-0.5 * boundaries * boundaries) * inv_sqrt_2pi
        masses = ndtr(boundaries[1:]) - ndtr(boundaries[:-1])
        updated = (density[:-1] - density[1:]) / masses
        if np.max(np.abs(updated - centroids)) < tolerance:
            centroids = updated
            break
        centroids = updated

    # Remove tiny numerical asymmetry so nearest-centroid ties are reproducible.
    return 0.5 * (centroids - centroids[::-1])


def nearest_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Quantize values using midpoint decision boundaries."""
    boundaries = 0.5 * (codebook[:-1] + codebook[1:])
    return codebook[np.searchsorted(boundaries, values)]


def randomized_hadamard(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply the orthonormal randomized Walsh-Hadamard transform H D."""
    out = np.asarray(values, dtype=np.float64) * signs
    size = out.size
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard transform dimension must be a power of two.")
    out = out.copy()
    width = 1
    while width < size:
        blocks = out.reshape(-1, width * 2)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return out / math.sqrt(size)


def pair_at_angle(dim: int, angle_deg: float, mode: str,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create either an isotropic or deliberately sparse unit-vector pair."""
    rho = math.cos(math.radians(angle_deg))
    orthogonal_weight = math.sqrt(max(0.0, 1.0 - rho * rho))
    if mode == "sparse":
        x = np.zeros(dim)
        y = np.zeros(dim)
        x[0] = 1.0
        y[0], y[1] = rho, orthogonal_weight
        return x, y

    x = rng.normal(size=dim)
    x /= np.linalg.norm(x)
    z = rng.normal(size=dim)
    z -= np.dot(z, x) * x
    z /= np.linalg.norm(z)
    return x, rho * x + orthogonal_weight * z


def angle_deg(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return float("nan")
    cosine = float(np.dot(x, y) / denominator)
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def int_quantizer(bits: int, dim: int, clip_sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    qmax = 2 ** (bits - 1) - 1
    step = clip_sigma / (qmax * math.sqrt(dim))
    return lambda values: np.clip(np.rint(values / step), -qmax, qmax)


def tq_quantizer(bits: int, dim: int) -> Callable[[np.ndarray], np.ndarray]:
    codebook = gaussian_lloyd_max_codebook(bits) / math.sqrt(dim)
    return lambda values: nearest_codebook(values, codebook)


def mean_distortion(angle: float, dim: int, trials: int, pair_mode: str,
                    quantizer: Callable[[np.ndarray], np.ndarray], transform: bool,
                    rng: np.random.Generator) -> tuple[float, float]:
    distortions = []
    for _ in range(trials):
        x, y = pair_at_angle(dim, angle, pair_mode, rng)
        if transform:
            signs = rng.choice((-1.0, 1.0), size=dim)
            x = randomized_hadamard(x, signs)
            y = randomized_hadamard(y, signs)
        distortions.append(angle_deg(quantizer(x), quantizer(y)) - angle)
    return float(np.mean(distortions)), float(np.std(distortions, ddof=1)) if trials > 1 else 0.0


def run(args: argparse.Namespace) -> None:
    angles = np.arange(args.angles_start, args.angles_stop + args.angles_step / 2,
                       args.angles_step)
    rng = np.random.default_rng(args.seed)
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(args.int_bits)))

    for color, bits in zip(colors, args.int_bits):
        quantizer = int_quantizer(bits, args.dim, args.clip_sigma)
        means = [mean_distortion(a, args.dim, args.trials, args.pair_mode,
                                 quantizer, False, rng)[0] for a in angles]
        ax.plot(angles, means, color=color, linewidth=1.5, label=f"INT{bits}")

    tq_colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.85, len(args.tq_bits)))
    for color, bits in zip(tq_colors, args.tq_bits):
        quantizer = tq_quantizer(bits, args.dim)
        transformed = [mean_distortion(a, args.dim, args.trials, args.pair_mode,
                                       quantizer, True, rng)[0] for a in angles]
        untransformed = [mean_distortion(a, args.dim, args.trials, args.pair_mode,
                                         quantizer, False, rng)[0] for a in angles]
        ax.plot(angles, transformed, color=color, linewidth=2.5, linestyle="--",
                label=f"TQ{bits} + randomized Hadamard")
        ax.plot(angles, untransformed, color=color, linewidth=2.0, linestyle=":",
                label=f"TQ{bits} codebook only (no transform)")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(title=f"Angular distortion ({args.pair_mode} pairs, d={args.dim})",
           xlabel="Original angle (degrees)", ylabel="Quantized angle - original angle (degrees)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--int-bits", type=int, nargs="+", default=list(range(3, 9)))
    parser.add_argument("--tq-bits", type=int, nargs="+", default=list(range(3, 9)))
    parser.add_argument("--pair-mode", choices=["sparse", "isotropic"], default="sparse")
    parser.add_argument("--angles-start", type=float, default=0.0)
    parser.add_argument("--angles-stop", type=float, default=90.0)
    parser.add_argument("--angles-step", type=float, default=3.0)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("outputs/turboquant_vs_int_angular_distortion.pdf"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dim < 2 or args.dim & (args.dim - 1):
        raise SystemExit("--dim must be a power of two and at least 2")
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    run(args)


if __name__ == "__main__":
    main()
