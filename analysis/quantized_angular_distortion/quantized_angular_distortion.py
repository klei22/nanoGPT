#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

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
        raise ValueError("Use bits >= 2.")
    return (2 ** (bits - 1)) - 1


def quantize_codes(values: np.ndarray, bits: int, scale: float) -> np.ndarray:
    qmax = qmax_for_bits(bits)
    return np.clip(np.rint(values / scale), -qmax, qmax).astype(np.float64)


def choose_scale(v: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> float:
    qmax = qmax_for_bits(bits)
    d = v.size
    if scale_mode == "fixed":
        full_scale = clip_sigma / math.sqrt(d)
    elif scale_mode == "std":
        full_scale = clip_sigma * float(np.std(v))
    elif scale_mode == "maxabs":
        full_scale = float(np.max(np.abs(v)))
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode}")
    return max(full_scale / qmax, np.finfo(float).tiny)


def quantize_vector(v: np.ndarray, bits: int, scale_mode: str, clip_sigma: float) -> np.ndarray:
    return quantize_codes(v, bits, choose_scale(v, bits, scale_mode, clip_sigma))


def quantized_angle_deg(x, y, bits, scale_mode, clip_sigma) -> float:
    qx = quantize_vector(x, bits, scale_mode, clip_sigma)
    qy = quantize_vector(y, bits, scale_mode, clip_sigma)
    nx, ny = float(np.linalg.norm(qx)), float(np.linalg.norm(qy))
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    c = float(np.dot(qx, qy) / (nx * ny))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def random_unit_pair_at_angle(dim: int, angle_deg: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_deg)
    rho = math.cos(theta)
    x = rng.normal(size=dim)
    x /= np.linalg.norm(x)

    z = rng.normal(size=dim)
    z -= np.dot(z, x) * x
    z /= np.linalg.norm(z)

    y = rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * z
    y /= np.linalg.norm(y)
    return x, y


def effective_tau(scale_mode: str, clip_sigma: float, dim: int) -> float:
    if scale_mode in {"fixed", "std"}:
        return float(clip_sigma)
    if scale_mode == "maxabs":
        return float(norm.ppf(1.0 - 1.0 / (2.0 * dim)))
    raise ValueError(f"Unknown scale_mode={scale_mode}")


def normal_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def quantizer_bins(qmax: int, delta: float):
    for k in range(-qmax, qmax + 1):
        if k == -qmax:
            yield k, -math.inf, (k + 0.5) * delta
        elif k == qmax:
            yield k, (k - 0.5) * delta, math.inf
        else:
            yield k, (k - 0.5) * delta, (k + 0.5) * delta


def code_from_standard_x(x, qmax, delta):
    return np.clip(np.rint(x / delta), -qmax, qmax).astype(np.float64)


def conditional_code_moments(mean, sd, qmax, delta, max_power=4):
    mean = np.asarray(mean, dtype=np.float64)
    out = np.zeros((max_power + 1, mean.size))
    out[0, :] = 1.0

    if sd < 1e-12:
        q = code_from_standard_x(mean, qmax, delta)
        for p in range(1, max_power + 1):
            out[p, :] = q ** p
        return out

    for k in range(-qmax, qmax + 1):
        if k == -qmax:
            lo, hi = -np.inf, (k + 0.5) * delta
        elif k == qmax:
            lo, hi = (k - 0.5) * delta, np.inf
        else:
            lo, hi = (k - 0.5) * delta, (k + 0.5) * delta

        pbin = ndtr((hi - mean) / sd) - ndtr((lo - mean) / sd)
        for p in range(1, max_power + 1):
            out[p, :] += (float(k) ** p) * pbin
    return out


def quantized_bivariate_code_moments(rho, qmax, delta, nodes_per_bin=96, tail_bound=12.0):
    rho = max(-1.0, min(1.0, float(rho)))
    sd = math.sqrt(max(0.0, 1.0 - rho * rho))
    lg_x, lg_w = leggauss(nodes_per_bin)
    moments = np.zeros((5, 5))

    for k, lo, hi in quantizer_bins(qmax, delta):
        a, b = max(lo, -tail_bound), min(hi, tail_bound)
        if not (a < b):
            continue

        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        x = mid + half * lg_x
        wx = half * lg_w * normal_pdf(x)

        y_mom = conditional_code_moments(rho * x, sd, qmax, delta, 4)
        integrated = np.sum(wx[None, :] * y_mom, axis=1)
        kpowers = np.array([float(k) ** p for p in range(5)])
        moments += kpowers[:, None] * integrated[None, :]

    return moments


def theory_for_angle(angle_deg, bits, dim, scale_mode, clip_sigma, quad_nodes=96) -> TheoryPoint:
    theta = math.radians(angle_deg)
    rho = max(-1.0, min(1.0, math.cos(theta)))
    qmax = qmax_for_bits(bits)
    tau = effective_tau(scale_mode, clip_sigma, dim)
    delta = tau / qmax

    M = quantized_bivariate_code_moments(
        rho, qmax, delta, nodes_per_bin=quad_nodes, tail_bound=max(12.0, tau + 8.0)
    )

    m_a = float(M[1, 1])
    m2 = float(M[2, 0])
    m4 = float(M[4, 0])
    e22 = float(M[2, 2])
    e31 = float(M[3, 1])
    e13 = float(M[1, 3])

    rho_q = max(-1.0, min(1.0, m_a / m2))
    theta_q = math.acos(rho_q)

    var_a = e22 - m_a * m_a
    var_b = m4 - m2 * m2
    cov_ab = e31 - m_a * m2
    cov_ac = e13 - m_a * m2
    cov_bc = e22 - m2 * m2

    cov = np.array([
        [var_a, cov_ab, cov_ac],
        [cov_ab, var_b, cov_bc],
        [cov_ac, cov_bc, var_b],
    ])

    grad = np.array([1.0 / m2, -m_a / (2.0 * m2 * m2), -m_a / (2.0 * m2 * m2)])
    var_rho_q = float(grad @ cov @ grad) / float(dim)
    var_theta_q = max(0.0, var_rho_q / max(1e-15, 1.0 - rho_q * rho_q))

    return TheoryPoint(
        angle_deg=angle_deg,
        bits=bits,
        rho_true=rho,
        rho_quant_theory=rho_q,
        quant_angle_deg_theory=math.degrees(theta_q),
        distortion_deg_theory=math.degrees(theta_q - theta),
        std_distortion_deg_delta=math.degrees(math.sqrt(var_theta_q)),
    )


def empirical_for_angle(angle_deg, bits, dim, trials, scale_mode, clip_sigma, rng) -> EmpiricalPoint:
    vals = []
    for _ in range(trials):
        x, y = random_unit_pair_at_angle(dim, angle_deg, rng)
        qa = quantized_angle_deg(x, y, bits, scale_mode, clip_sigma)
        if np.isfinite(qa):
            vals.append(qa - angle_deg)

    arr = np.asarray(vals)
    return EmpiricalPoint(
        angle_deg=angle_deg,
        bits=bits,
        mean_distortion_deg=float(np.mean(arr)) if arr.size else float("nan"),
        std_distortion_deg=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        valid_trials=int(arr.size),
    )


def parse_angles(args):
    if args.angles:
        return [float(a) for a in args.angles]
    return [float(a) for a in np.arange(args.angles_start, args.angles_stop + 0.5 * args.angles_step, args.angles_step)]


def write_csv(path, theory, empirical, args):
    emp_map = {(e.bits, e.angle_deg): e for e in empirical}
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "bits", "angle_deg", "rho_true", "rho_quant_theory",
            "quant_angle_deg_theory", "distortion_deg_theory",
            "std_distortion_deg_delta_method", "mean_distortion_deg_empirical",
            "std_distortion_deg_empirical", "valid_trials",
            "dim", "scale_mode", "clip_sigma",
        ])
        for t in theory:
            e = emp_map.get((t.bits, t.angle_deg))
            w.writerow([
                t.bits, t.angle_deg, t.rho_true, t.rho_quant_theory,
                t.quant_angle_deg_theory, t.distortion_deg_theory,
                t.std_distortion_deg_delta,
                e.mean_distortion_deg if e else "",
                e.std_distortion_deg if e else "",
                e.valid_trials if e else "",
                args.dim, args.scale_mode, args.clip_sigma,
            ])


def setup_pub_style():
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


def plot_results(path, bits_list, angles, theory, empirical, args):
    setup_pub_style()

    t_by_bits = {b: [p for p in theory if p.bits == b] for b in bits_list}
    e_by_bits = {b: [p for p in empirical if p.bits == b] for b in bits_list}
    colors = {3: "#1f77b4", 4: "#2ca02c", 5: "#9467bd", 6: "#8c564b", 7: "#e377c2", 8: "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
    ax.axvspan(20, 30, color="gray", alpha=0.12, label=r"$20^\circ$--$30^\circ$ regime")

    for bits in bits_list:
        ts = sorted(t_by_bits[bits], key=lambda p: p.angle_deg)
        es = sorted(e_by_bits[bits], key=lambda p: p.angle_deg)

        ax.plot(
            [p.angle_deg for p in ts],
            [p.distortion_deg_theory for p in ts],
            color=colors.get(bits),
            label=fr"INT{bits} transfer",
        )

        if es:
            ax.errorbar(
                [p.angle_deg for p in es],
                [p.mean_distortion_deg for p in es],
                yerr=[p.std_distortion_deg for p in es],
                fmt="o",
                color=colors.get(bits),
                markersize=2.4,
                capsize=1.6,
                elinewidth=0.75,
                alpha=0.45,
                label=fr"INT{bits} MC",
            )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 10))
    ax.set_xlabel(r"Original angle $\theta$ (degrees)")
    ax.set_ylabel(r"Angular distortion $\arccos(C_b(\cos\theta))-\theta$ (degrees)")
    ax.set_title("Angle-dependent distortion from low-bit symmetric quantization")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_sweep_by_bit(path, bit, dim_to_theory: Dict[int, List[TheoryPoint]], dim_to_empirical: Dict[int, List[EmpiricalPoint]]):
    setup_pub_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0.0, linewidth=1, linestyle="--", color="black", alpha=0.5)
    ax.axvspan(20, 30, color="gray", alpha=0.12, label=r"$20^\circ$--$30^\circ$ regime")

    cmap = plt.get_cmap("viridis")
    dims = sorted(dim_to_theory.keys())

    for idx, dim in enumerate(dims):
        color = cmap(idx / max(1, len(dims) - 1))
        ts = sorted([p for p in dim_to_theory[dim] if p.bits == bit], key=lambda p: p.angle_deg)
        es = sorted([p for p in dim_to_empirical.get(dim, []) if p.bits == bit], key=lambda p: p.angle_deg)

        ax.plot(
            [p.angle_deg for p in ts],
            [p.distortion_deg_theory for p in ts],
            color=color,
            label=fr"$d={dim}$ transfer",
        )

        if es:
            ax.errorbar(
                [p.angle_deg for p in es],
                [p.mean_distortion_deg for p in es],
                yerr=[p.std_distortion_deg for p in es],
                fmt="o",
                color=color,
                markersize=2.0,
                capsize=1.3,
                elinewidth=0.65,
                alpha=0.35,
            )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 10))
    ax.set_xlabel(r"Original angle $\theta$ (degrees)")
    ax.set_ylabel(r"Angular distortion $\arccos(C_b(\cos\theta))-\theta$ (degrees)")
    ax.set_title(fr"Dimension sweep for INT{bit} quantization")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_arcsine_alignment(output: Path, n=2_000_000, seed=0):
    setup_pub_style()
    rng = np.random.default_rng(seed)
    rhos = np.linspace(-0.99, 0.99, 101)
    mc = []

    for rho in rhos:
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        y = rho * x + np.sqrt(1.0 - rho ** 2) * z
        mc.append(np.mean(np.sign(x) * np.sign(y)))

    theory = (2.0 / np.pi) * np.arcsin(rhos)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(rhos, theory, label=r"$\frac{2}{\pi}\arcsin(\rho)$")
    ax.scatter(rhos, mc, s=10, alpha=0.55, label="Gaussian MC")
    ax.set_xlabel(r"True Gaussian correlation $\rho$")
    ax.set_ylabel("Sign-quantized correlation")
    ax.set_title("Arcsine-law alignment")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="Optional dimension sweep, e.g. --dims 512 1024 2048 4096")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--angles", type=float, nargs="*", default=None)
    parser.add_argument("--angles-start", type=float, default=0.0)
    parser.add_argument("--angles-stop", type=float, default=90.0)
    parser.add_argument("--angles-step", type=float, default=1.0)
    parser.add_argument("--scale-mode", choices=["fixed", "std", "maxabs"], default="fixed")
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--quad-nodes", type=int, default=96)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("gaussian_transfer_distortion.pdf"))
    parser.add_argument("--csv", type=Path, default=Path("gaussian_transfer_distortion.csv"))
    parser.add_argument("--no-monte-carlo", action="store_true")
    parser.add_argument("--no-arcsine-plot", action="store_true")
    parser.add_argument("--no-dim-sweep-plots", action="store_true")
    args = parser.parse_args()

    dims = args.dims if args.dims is not None else [args.dim]
    if any(d < 3 for d in dims):
        raise ValueError("all dimensions must be at least 3")

    angles = parse_angles(args)

    all_theory: Dict[int, List[TheoryPoint]] = {}
    all_empirical: Dict[int, List[EmpiricalPoint]] = {}

    for dim in dims:
        print(f"\n=== Running dimension d={dim} ===")
        args.dim = dim
        rng = np.random.default_rng(args.seed)

        theory: List[TheoryPoint] = []
        empirical: List[EmpiricalPoint] = []

        print("Computing theory...")
        for bits in args.bits:
            for angle in angles:
                theory.append(theory_for_angle(angle, bits, dim, args.scale_mode, args.clip_sigma, args.quad_nodes))

        if not args.no_monte_carlo:
            print("Running Monte Carlo...")
            for bits in args.bits:
                for angle in angles:
                    empirical.append(empirical_for_angle(angle, bits, dim, args.trials, args.scale_mode, args.clip_sigma, rng))

        all_theory[dim] = theory
        all_empirical[dim] = empirical

        out_path = args.output.with_name(f"{args.output.stem}_d{dim}{args.output.suffix}")
        csv_path = args.csv.with_name(f"{args.csv.stem}_d{dim}{args.csv.suffix}")

        print(f"Writing CSV: {csv_path}")
        write_csv(csv_path, theory, empirical, args)

        print(f"Writing plot: {out_path}")
        plot_results(out_path, args.bits, angles, theory, empirical, args)

    if len(dims) > 1 and not args.no_dim_sweep_plots:
        for bits in args.bits:
            sweep_path = args.output.with_name(f"{args.output.stem}_INT{bits}_dimension_sweep{args.output.suffix}")
            print(f"Writing dimension sweep plot: {sweep_path}")
            plot_dimension_sweep_by_bit(sweep_path, bits, all_theory, all_empirical)

    if not args.no_arcsine_plot:
        arcsine_path = args.output.parent / "arcsine_alignment.pdf"
        print(f"Writing arcsine-law plot: {arcsine_path}")
        plot_arcsine_alignment(arcsine_path)

    print("Done.")


if __name__ == "__main__":
    main()
