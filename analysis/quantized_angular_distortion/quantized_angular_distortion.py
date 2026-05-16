#!/usr/bin/env python3
"""
Quantized angular distortion for random normalized vector pairs.

This script overlays:
  1) a high-dimensional Gaussian theory prediction for a symmetric low-bit quantizer, and
  2) Monte Carlo mean +/- one standard deviation for random unit-vector pairs at fixed angles.

Default quantizer:
  signed symmetric mid-tread integer codes [-qmax, ..., 0, ..., qmax]
  qmax = 2^(bits-1) - 1
  code = clip(round(x / scale), -qmax, qmax)

Theory model:
  If x,y are high-dimensional unit vectors at angle theta, then each scaled coordinate pair
  (sqrt(d)*x_i, sqrt(d)*y_i) is approximately bivariate normal with correlation rho=cos(theta).
  The predicted quantized cosine is

      E[Q(X)Q(Y)] / E[Q(X)^2]

  where (X,Y) are standard normal with corr(X,Y)=rho. The angular distortion is

      arccos(predicted_quantized_cosine) - theta.

Dependencies: numpy, scipy, matplotlib

Example:
  python quantized_angular_distortion.py --dim 4096 --trials 500 --bits 3 4 5 \
      --angles-start 5 --angles-stop 85 --angles-step 1 \
      --scale-mode fixed --clip-sigma 3.0 --output distortion.png --csv distortion.csv

For a rough match to per-vector absmax quantization, try:
  python quantized_angular_distortion.py --scale-mode maxabs --dim 4096
The theory then uses an effective Gaussian max threshold tau ~= Phi^{-1}(1 - 1/(2d)).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from scipy.stats import norm


@dataclass(frozen=True)
class TheoryPoint:
    angle_deg: float
    bits: int
    rho_true: float
    rho_quant_theory: float
    quant_angle_deg_theory: float
    distortion_deg_theory: float
    std_distortion_deg_delta: float


@dataclass(frozen=True)
class EmpiricalPoint:
    angle_deg: float
    bits: int
    mean_distortion_deg: float
    std_distortion_deg: float
    valid_trials: int


def qmax_for_bits(bits: int) -> int:
    if bits < 2:
        raise ValueError("Use bits >= 2 for this signed symmetric integer quantizer.")
    return (2 ** (bits - 1)) - 1


def quantize_codes(values: np.ndarray, bits: int, scale: float) -> np.ndarray:
    """Return symmetric signed integer quantization codes, not dequantized values."""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Bad quantization scale: {scale}")
    qmax = qmax_for_bits(bits)
    codes = np.rint(values / scale)
    codes = np.clip(codes, -qmax, qmax)
    return codes.astype(np.float64)


def choose_scale(v: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> float:
    """
    Scale for unit-vector coordinates.

    fixed:  assumes coordinates have std ~ 1/sqrt(d), uses clip_sigma/sqrt(d) as full-scale.
    std:    uses clip_sigma * empirical std(v) as full-scale.
    maxabs: uses max(abs(v)) as full-scale, common in per-vector/per-tensor absmax quantization.
    """
    qmax = qmax_for_bits(bits)
    d = v.size
    if scale_mode == "fixed":
        full_scale = clip_sigma / math.sqrt(d)
    elif scale_mode == "std":
        full_scale = clip_sigma * float(np.std(v))
    elif scale_mode == "maxabs":
        full_scale = float(np.max(np.abs(v)))
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode!r}")
    # Avoid an all-zero pathological vector or zero scale.
    return max(full_scale / qmax, np.finfo(float).tiny)


def quantize_vector(v: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> np.ndarray:
    scale = choose_scale(v, bits, scale_mode, clip_sigma)
    return quantize_codes(v, bits, scale)


def quantized_angle_deg(x: np.ndarray, y: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> float:
    qx = quantize_vector(x, bits, scale_mode, clip_sigma)
    qy = quantize_vector(y, bits, scale_mode, clip_sigma)
    nx = float(np.linalg.norm(qx))
    ny = float(np.linalg.norm(qy))
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    c = float(np.dot(qx, qy) / (nx * ny))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def random_unit_pair_at_angle(dim: int, angle_deg: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Create x,y on the unit sphere with exactly the requested angle."""
    theta = math.radians(angle_deg)
    rho = math.cos(theta)
    x = rng.normal(size=dim)
    x /= np.linalg.norm(x)

    z = rng.normal(size=dim)
    z -= np.dot(z, x) * x
    z_norm = np.linalg.norm(z)
    # Very unlikely fallback for numerical degeneracy.
    if z_norm == 0.0:
        z = rng.normal(size=dim)
        z -= np.dot(z, x) * x
        z_norm = np.linalg.norm(z)
    z /= z_norm

    y = rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * z
    y /= np.linalg.norm(y)
    return x, y


def effective_tau(scale_mode: str, clip_sigma: float, dim: int) -> float:
    """
    Full-scale threshold tau in standardized coordinates X=sqrt(d)*x_i.

    fixed/std: tau = clip_sigma.
    maxabs:    tau approximates max_i |Z_i| for Z_i~N(0,1), using a high quantile.
               This is only an approximation because the actual maxabs scale is random.
    """
    if scale_mode in {"fixed", "std"}:
        return float(clip_sigma)
    if scale_mode == "maxabs":
        # P(max |Z_i| <= tau) = (2Phi(tau)-1)^dim.
        # A simple effective value is close to the expected/typical max.
        # The 1 - 1/(2d) normal quantile is a standard asymptotic approximation.
        return float(norm.ppf(1.0 - 1.0 / (2.0 * dim)))
    raise ValueError(f"Unknown scale_mode={scale_mode!r}")


def code_from_standard_x(x: np.ndarray, qmax: int, delta: float) -> np.ndarray:
    """Quantization code for standardized Gaussian coordinate x."""
    return np.clip(np.rint(x / delta), -qmax, qmax).astype(np.float64)


def conditional_code_moments(mean: np.ndarray, sd: float, qmax: int, delta: float, max_power: int = 4) -> np.ndarray:
    """
    For Y ~ N(mean, sd^2), return E[Q(Y)^p | mean] for p=0..max_power.
    mean can be a vector. Output shape: (max_power+1, len(mean)).
    """
    mean = np.asarray(mean, dtype=np.float64)
    out = np.zeros((max_power + 1, mean.size), dtype=np.float64)

    if sd < 1e-12:
        q = code_from_standard_x(mean, qmax, delta)
        out[0, :] = 1.0
        for p in range(1, max_power + 1):
            out[p, :] = q ** p
        return out

    out[0, :] = 1.0
    for k in range(-qmax, qmax + 1):
        if k == -qmax:
            lo = -np.inf
            hi = (k + 0.5) * delta
        elif k == qmax:
            lo = (k - 0.5) * delta
            hi = np.inf
        else:
            lo = (k - 0.5) * delta
            hi = (k + 0.5) * delta

        pbin = ndtr((hi - mean) / sd) - ndtr((lo - mean) / sd)
        for power in range(1, max_power + 1):
            out[power, :] += (float(k) ** power) * pbin

    return out


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def quantizer_bins(qmax: int, delta: float) -> List[Tuple[int, float, float]]:
    """Return (code, low, high) bins in standardized-coordinate units."""
    bins: List[Tuple[int, float, float]] = []
    for k in range(-qmax, qmax + 1):
        if k == -qmax:
            lo, hi = -math.inf, (k + 0.5) * delta
        elif k == qmax:
            lo, hi = (k - 0.5) * delta, math.inf
        else:
            lo, hi = (k - 0.5) * delta, (k + 0.5) * delta
        bins.append((k, lo, hi))
    return bins


def quantized_bivariate_code_moments(
    rho: float,
    qmax: int,
    delta: float,
    nodes_per_bin: int = 96,
    tail_bound: float = 12.0,
) -> np.ndarray:
    """
    Compute moments M[p,r] = E[Q(X)^p Q(Y)^r] for p,r=0..4,
    where (X,Y) are standard normal with corr(X,Y)=rho.

    The integration is one-dimensional:
      Y | X=x ~ N(rho*x, 1-rho^2),
    and X is integrated piecewise over Q(X)'s bins. Piecewise integration is much more
    accurate than global Gauss-Hermite quadrature for high correlations, because quantizer
    thresholds create narrow boundary layers.
    """
    rho = max(-1.0, min(1.0, float(rho)))
    sd = math.sqrt(max(0.0, 1.0 - rho * rho))
    lg_x, lg_w = leggauss(nodes_per_bin)

    moments = np.zeros((5, 5), dtype=np.float64)
    for k, lo, hi in quantizer_bins(qmax, delta):
        # The infinite tails are saturated bins. Truncating at +/- tail_bound loses negligible
        # standard-normal mass when tail_bound>=10.
        a = max(lo, -tail_bound)
        b = min(hi, tail_bound)
        if not (a < b):
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        x = mid + half * lg_x
        wx = half * lg_w * normal_pdf(x)
        y_mom = conditional_code_moments(rho * x, sd, qmax, delta, max_power=4)
        # y_mom shape: (5, nodes_per_bin)
        integrated_y_mom = np.sum(wx[None, :] * y_mom, axis=1)
        kpowers = np.array([float(k) ** p for p in range(5)], dtype=np.float64)
        moments += kpowers[:, None] * integrated_y_mom[None, :]

    return moments


def theory_for_angle(
    angle_deg: float,
    bits: int,
    dim: int,
    scale_mode: str,
    clip_sigma: float,
    quad_nodes: int = 96,
) -> TheoryPoint:
    """High-dimensional Gaussian theory + delta-method angular std."""
    theta = math.radians(angle_deg)
    rho = math.cos(theta)
    rho = max(-1.0, min(1.0, rho))

    qmax = qmax_for_bits(bits)
    tau = effective_tau(scale_mode, clip_sigma, dim)
    delta = tau / qmax

    # Moment matrix M[p,r] = E[Q(X)^p Q(Y)^r].
    tail_bound = max(12.0, tau + 8.0)
    M = quantized_bivariate_code_moments(
        rho=rho,
        qmax=qmax,
        delta=delta,
        nodes_per_bin=quad_nodes,
        tail_bound=tail_bound,
    )

    # Raw moments needed for predicted cosine and delta-method variance.
    m_a = float(M[1, 1])   # E[QX QY]
    m2 = float(M[2, 0])    # E[QX^2]
    m4 = float(M[4, 0])    # E[QX^4]
    e22 = float(M[2, 2])   # E[QX^2 QY^2]
    e31 = float(M[3, 1])   # E[QX^3 QY]
    e13 = float(M[1, 3])   # E[QX QY^3]

    rho_q = m_a / m2
    rho_q = max(-1.0, min(1.0, rho_q))
    theta_q = math.acos(rho_q)
    distortion_deg = math.degrees(theta_q - theta)

    # Delta-method variance for cosine = sum A_i / sqrt(sum B_i sum C_i)
    # A=QX QY, B=QX^2, C=QY^2. Coordinates are treated as iid in the high-d model.
    var_a = e22 - m_a * m_a
    var_b = m4 - m2 * m2
    var_c = var_b
    cov_ab = e31 - m_a * m2
    cov_ac = e13 - m_a * m2
    cov_bc = e22 - m2 * m2
    cov = np.array(
        [
            [var_a, cov_ab, cov_ac],
            [cov_ab, var_b, cov_bc],
            [cov_ac, cov_bc, var_c],
        ],
        dtype=np.float64,
    )
    grad = np.array([1.0 / m2, -m_a / (2.0 * m2 * m2), -m_a / (2.0 * m2 * m2)], dtype=np.float64)
    var_rho_q = float(grad @ cov @ grad) / float(dim)
    denom = max(1e-15, 1.0 - rho_q * rho_q)
    var_theta_q = max(0.0, var_rho_q / denom)
    std_distortion_deg = math.degrees(math.sqrt(var_theta_q))

    return TheoryPoint(
        angle_deg=angle_deg,
        bits=bits,
        rho_true=rho,
        rho_quant_theory=rho_q,
        quant_angle_deg_theory=math.degrees(theta_q),
        distortion_deg_theory=distortion_deg,
        std_distortion_deg_delta=std_distortion_deg,
    )

def empirical_for_angle(
    angle_deg: float,
    bits: int,
    dim: int,
    trials: int,
    scale_mode: str,
    clip_sigma: float,
    rng: np.random.Generator,
) -> EmpiricalPoint:
    distortions = []
    for _ in range(trials):
        x, y = random_unit_pair_at_angle(dim, angle_deg, rng)
        qa = quantized_angle_deg(x, y, bits, scale_mode, clip_sigma)
        if np.isfinite(qa):
            distortions.append(qa - angle_deg)

    arr = np.asarray(distortions, dtype=np.float64)
    if arr.size == 0:
        return EmpiricalPoint(angle_deg, bits, float("nan"), float("nan"), 0)
    return EmpiricalPoint(
        angle_deg=angle_deg,
        bits=bits,
        mean_distortion_deg=float(np.mean(arr)),
        std_distortion_deg=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        valid_trials=int(arr.size),
    )


def parse_angles(args: argparse.Namespace) -> List[float]:
    if args.angles:
        return [float(a) for a in args.angles]
    # Inclusive-ish stop for integer-like ranges.
    values = np.arange(args.angles_start, args.angles_stop + 0.5 * args.angles_step, args.angles_step)
    return [float(a) for a in values]


def write_csv(path: Path, theory: List[TheoryPoint], empirical: List[EmpiricalPoint], args: argparse.Namespace) -> None:
    emp_map = {(e.bits, e.angle_deg): e for e in empirical}
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bits",
                "angle_deg",
                "rho_true",
                "rho_quant_theory",
                "quant_angle_deg_theory",
                "distortion_deg_theory",
                "std_distortion_deg_delta_method",
                "mean_distortion_deg_empirical",
                "std_distortion_deg_empirical",
                "valid_trials",
                "dim",
                "scale_mode",
                "clip_sigma",
            ]
        )
        for t in theory:
            e = emp_map.get((t.bits, t.angle_deg))
            writer.writerow(
                [
                    t.bits,
                    t.angle_deg,
                    t.rho_true,
                    t.rho_quant_theory,
                    t.quant_angle_deg_theory,
                    t.distortion_deg_theory,
                    t.std_distortion_deg_delta,
                    e.mean_distortion_deg if e else "",
                    e.std_distortion_deg if e else "",
                    e.valid_trials if e else "",
                    args.dim,
                    args.scale_mode,
                    args.clip_sigma,
                ]
            )


def plot_results(path, bits_list, angles, theory, empirical, args):
    t_by_bits = {b: [p for p in theory if p.bits == b] for b in bits_list}
    e_by_bits = {b: [p for p in empirical if p.bits == b] for b in bits_list}

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "lines.linewidth": 2.2,
    })

    colors = {
        3: "#1f77b4",
        4: "#2ca02c",
        5: "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.axhline(0.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
    ax.axvspan(
        20, 30,
        color="gray",
        alpha=0.12,
        label=r"$20^\circ$--$30^\circ$ regime",
    )

    for bits in bits_list:
        ts = sorted(t_by_bits[bits], key=lambda p: p.angle_deg)
        es = sorted(e_by_bits[bits], key=lambda p: p.angle_deg)

        x_t = [p.angle_deg for p in ts]
        y_t = [p.distortion_deg_theory for p in ts]

        ax.plot(
            x_t,
            y_t,
            color=colors.get(bits, None),
            label=fr"INT{bits} predicted",
        )

        if es:
            x_e = [p.angle_deg for p in es]
            y_e = [p.mean_distortion_deg for p in es]
            yerr_e = [p.std_distortion_deg for p in es]

            ax.errorbar(
                x_e,
                y_e,
                yerr=yerr_e,
                fmt="o",
                color=colors.get(bits, None),
                markersize=2.4,
                capsize=1.6,
                elinewidth=0.75,
                alpha=0.45,
                label=fr"INT{bits} MC",
            )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 10))

    ax.set_xlabel(r"Original angle $\theta$ (degrees)")
    ax.set_ylabel(
        r"Angular distortion $\arccos(C_b(\cos\theta))-\theta$ (degrees)"
    )
    ax.set_title("Angle-dependent distortion from low-bit symmetric quantization")

    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
def plot_arcsine_alignment(
    output: Path = Path("arcsine_alignment.png"),
    n: int = 2_000_000,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    rhos = np.linspace(-0.99, 0.99, 101)
    mc = []

    for rho in rhos:
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        y = rho * x + np.sqrt(1.0 - rho**2) * z
        mc.append(np.mean(np.sign(x) * np.sign(y)))

    theory = (2.0 / np.pi) * np.arcsin(rhos)

    plt.figure(figsize=(7, 5))
    plt.plot(rhos, theory, label=r"$\frac{2}{\pi}\arcsin(\rho)$", linewidth=2)
    plt.scatter(rhos, mc, s=10, alpha=0.55, label="Gaussian Monte Carlo")
    plt.xlabel(r"True Gaussian correlation $\rho$")
    plt.ylabel("Sign-quantized correlation")
    plt.title("Arcsine-law alignment for sign-quantized Gaussian pairs")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096, help="Vector dimension.")
    parser.add_argument("--trials", type=int, default=300, help="Monte Carlo trials per angle per bit depth.")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5], help="Bit depths, e.g. 3 4 5.")
    parser.add_argument("--angles", type=float, nargs="*", default=None, help="Explicit angles in degrees.")
    parser.add_argument("--angles-start", type=float, default=5.0)
    parser.add_argument("--angles-stop", type=float, default=85.0)
    parser.add_argument("--angles-step", type=float, default=1.0)
    parser.add_argument("--scale-mode", choices=["fixed", "std", "maxabs"], default="fixed")
    parser.add_argument("--clip-sigma", type=float, default=3.0, help="Full-scale threshold in Gaussian sigma units for fixed/std modes.")
    parser.add_argument("--quad-nodes", type=int, default=96, help="Gauss-Legendre nodes per quantizer bin for theory.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("quantized_angular_distortion.png"))
    parser.add_argument("--csv", type=Path, default=Path("quantized_angular_distortion.csv"))
    parser.add_argument("--no-monte-carlo", action="store_true", help="Only plot theory curves; skip empirical simulation.")
    args = parser.parse_args()

    if args.dim < 3:
        raise ValueError("dim must be at least 3.")
    if args.trials < 1 and not args.no_monte_carlo:
        raise ValueError("trials must be >= 1 unless --no-monte-carlo is used.")

    angles = parse_angles(args)
    rng = np.random.default_rng(args.seed)

    theory: List[TheoryPoint] = []
    empirical: List[EmpiricalPoint] = []

    print("Computing theory...")
    for bits in args.bits:
        for angle in angles:
            theory.append(
                theory_for_angle(
                    angle_deg=angle,
                    bits=bits,
                    dim=args.dim,
                    scale_mode=args.scale_mode,
                    clip_sigma=args.clip_sigma,
                    quad_nodes=args.quad_nodes,
                )
            )

    if not args.no_monte_carlo:
        print("Running Monte Carlo...")
        for bits in args.bits:
            for angle in angles:
                empirical.append(
                    empirical_for_angle(
                        angle_deg=angle,
                        bits=bits,
                        dim=args.dim,
                        trials=args.trials,
                        scale_mode=args.scale_mode,
                        clip_sigma=args.clip_sigma,
                        rng=rng,
                    )
                )

    print(f"Writing CSV: {args.csv}")
    write_csv(args.csv, theory, empirical, args)
    print(f"Writing plot: {args.output}")

    plot_results(args.output, args.bits, angles, theory, empirical, args)

    arcsine_output = args.output.parent / "arcsine_alignment.png"
    print(f"Writing arcsine-law plot: {arcsine_output}")
    plot_arcsine_alignment(arcsine_output)

    print("Done.")


if __name__ == "__main__":
    main()
