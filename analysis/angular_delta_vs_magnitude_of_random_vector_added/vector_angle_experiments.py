import argparse
from typing import Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def generate_normalized_vector(dim: int, init_std: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a Gaussian random vector and L2-normalize it."""
    v = rng.normal(loc=0.0, scale=init_std, size=dim)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        # Extremely unlikely, but try once more
        v = rng.normal(loc=0.0, scale=init_std, size=dim)
        norm = np.linalg.norm(v)
        if norm == 0.0:
            # Fallback to unit vector on first coordinate
            e0 = np.zeros(dim, dtype=float)
            e0[0] = 1.0
            return e0
    return v / norm


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Return the angle in degrees between two vectors assumed to be L2-normalized."""
    cos_theta = float(np.dot(u, v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def run_noise_experiment(
    dimensions: Sequence[int],
    noise_stds: Sequence[float],
    num_samples: int,
    base_init_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Experiment 1:
    - For each dimension and noise stddev, sample `num_samples` base vectors.
    - Perturb each base vector with Gaussian noise of given stddev, renormalize.
    - Measure angle between original and perturbed vectors.
    Returns:
        means: shape (len(dimensions), len(noise_stds))
        stds:  shape (len(dimensions), len(noise_stds))
    """
    rng = np.random.default_rng(seed)
    dimensions = list(dimensions)
    noise_stds = np.array(list(noise_stds), dtype=float)

    means = np.zeros((len(dimensions), len(noise_stds)), dtype=float)
    stds = np.zeros_like(means)

    for i_dim, dim in enumerate(dimensions):
        angle_records = {std: [] for std in noise_stds}
        for _ in range(num_samples):
            base = generate_normalized_vector(dim, base_init_std, rng)
            for std in noise_stds:
                noise = rng.normal(loc=0.0, scale=float(std), size=dim)
                perturbed = base + noise
                norm_p = np.linalg.norm(perturbed)
                if norm_p == 0.0:
                    perturbed_normed = base
                else:
                    perturbed_normed = perturbed / norm_p
                angle_deg = angle_between_deg(base, perturbed_normed)
                angle_records[std].append(angle_deg)

        for j_std, std in enumerate(noise_stds):
            arr = np.asarray(angle_records[std], dtype=float)
            means[i_dim, j_std] = arr.mean()
            stds[i_dim, j_std] = arr.std(ddof=1)

    return means, stds


def run_alpha_experiment(
    dimensions: Sequence[int],
    alphas: Sequence[float],
    num_samples: int,
    base_init_std: float,
    delta_init_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Experiment 2:
    - For each dimension and alpha, sample `num_samples` pairs (base, delta).
    - base ~ Gaussian(init_std=base_init_std), L2-normalized
    - delta ~ Gaussian(init_std=delta_init_std), L2-normalized
    - scaled_delta = alpha * delta
    - final = base + scaled_delta, then L2-normalized
    - measure angle between base and final.
    Returns:
        means: shape (len(dimensions), len(alphas))
        stds:  shape (len(dimensions), len(alphas))
    """
    rng = np.random.default_rng(seed)
    dimensions = list(dimensions)
    alphas = np.array(list(alphas), dtype=float)

    means = np.zeros((len(dimensions), len(alphas)), dtype=float)
    stds = np.zeros_like(means)

    for i_dim, dim in enumerate(dimensions):
        angle_records = {alpha: [] for alpha in alphas}
        for _ in range(num_samples):
            base = generate_normalized_vector(dim, base_init_std, rng)
            delta_unit = generate_normalized_vector(dim, delta_init_std, rng)
            for alpha in alphas:
                final = base + float(alpha) * delta_unit
                norm_f = np.linalg.norm(final)
                if norm_f == 0.0:
                    final_normed = base
                else:
                    final_normed = final / norm_f
                angle_deg = angle_between_deg(base, final_normed)
                angle_records[alpha].append(angle_deg)

        for j_alpha, alpha in enumerate(alphas):
            arr = np.asarray(angle_records[alpha], dtype=float)
            means[i_dim, j_alpha] = arr.mean()
            stds[i_dim, j_alpha] = arr.std(ddof=1)

    return means, stds


def plot_noise_matplotlib(
    dimensions: Sequence[int],
    noise_stds: Sequence[float],
    means: np.ndarray,
    stds: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Static matplotlib plot for Experiment 1 (noise std sweep)."""
    noise_stds = np.array(list(noise_stds), dtype=float)
    plt.figure(figsize=(8, 6))

    for i_dim, dim in enumerate(dimensions):
        plt.errorbar(
            noise_stds,
            means[i_dim],
            yerr=stds[i_dim],
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"d = {dim}",
        )

    plt.xscale("log")
    plt.xlabel("Perturbation standard deviation")
    plt.ylabel("Mean angular difference (degrees)")
    plt.title("Experiment 1: Angular deviation vs noise stddev")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_alpha_matplotlib(
    dimensions: Sequence[int],
    alphas: Sequence[float],
    means: np.ndarray,
    stds: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Static matplotlib plot for Experiment 2 (alpha sweep)."""
    alphas = np.array(list(alphas), dtype=float)
    plt.figure(figsize=(8, 6))

    for i_dim, dim in enumerate(dimensions):
        plt.errorbar(
            alphas,
            means[i_dim],
            yerr=stds[i_dim],
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"d = {dim}",
        )

    plt.xlabel("alpha (scale of normalized delta vector)")
    plt.ylabel("Mean angular difference (degrees)")
    plt.title("Experiment 2: Angular deviation vs alpha")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_noise_html(
    dimensions: Sequence[int],
    noise_stds: Sequence[float],
    means: np.ndarray,
    stds: np.ndarray,
    html_path: str,
) -> None:
    """Interactive HTML plot (Plotly) for Experiment 1."""
    # Local import so the rest of the script works even if Plotly isn't installed
    import plotly.graph_objects as go
    import plotly.io as pio

    noise_stds = np.array(list(noise_stds), dtype=float)

    fig = go.Figure()
    for i_dim, dim in enumerate(dimensions):
        fig.add_trace(
            go.Scatter(
                x=noise_stds,
                y=means[i_dim],
                mode="lines+markers",
                name=f"d = {dim}",
                error_y=dict(
                    type="data",
                    array=stds[i_dim],
                    visible=True,
                ),
            )
        )

    fig.update_xaxes(type="log", title_text="Perturbation standard deviation")
    fig.update_yaxes(title_text="Mean angular difference (degrees)")
    fig.update_layout(
        title="Experiment 1: Angular deviation vs noise stddev",
        hovermode="x unified",
    )

    pio.write_html(fig, file=html_path, auto_open=False)


def plot_alpha_html(
    dimensions: Sequence[int],
    alphas: Sequence[float],
    means: np.ndarray,
    stds: np.ndarray,
    html_path: str,
) -> None:
    """Interactive HTML plot (Plotly) for Experiment 2."""
    import plotly.graph_objects as go
    import plotly.io as pio

    alphas = np.array(list(alphas), dtype=float)

    fig = go.Figure()
    for i_dim, dim in enumerate(dimensions):
        fig.add_trace(
            go.Scatter(
                x=alphas,
                y=means[i_dim],
                mode="lines+markers",
                name=f"d = {dim}",
                error_y=dict(
                    type="data",
                    array=stds[i_dim],
                    visible=True,
                ),
            )
        )

    fig.update_xaxes(title_text="alpha (scale of normalized delta vector)")
    fig.update_yaxes(title_text="Mean angular difference (degrees)")
    fig.update_layout(
        title="Experiment 2: Angular deviation vs alpha",
        hovermode="x unified",
    )

    pio.write_html(fig, file=html_path, auto_open=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiments on angular deviation between L2-normalized vectors "
            "under Gaussian perturbations and scaled normalized deltas."
        )
    )
    parser.add_argument(
        "--base-std",
        type=float,
        default=0.02,
        help="Stddev for base vector Gaussian initialization (default: 0.02).",
    )
    parser.add_argument(
        "--delta-std",
        type=float,
        default=0.02,
        help=(
            "Stddev for delta vector Gaussian initialization in the alpha sweep "
            "(Experiment 2). Default: 0.02."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples per (dimension, std/alpha) configuration (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="vector_angle",
        help="Prefix for output PNG/HTML files (default: vector_angle).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Dimensions from 64 to 1024 in powers of 2
    dimensions = [64, 128, 256, 512, 1024]

    # Stddevs to sweep in Experiment 1
    noise_stds = [2e-5, 2e-4, 2e-3, 2e-2, 2e-1]

    # Alpha values to sweep in Experiment 2
    alphas = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0

    # -------- Experiment 1: noise stddev sweep --------
    means_noise, stds_noise = run_noise_experiment(
        dimensions=dimensions,
        noise_stds=noise_stds,
        num_samples=args.num_samples,
        base_init_std=args.base_std,
        seed=args.seed,
    )

    print("Experiment 1: Noise stddev sweep")
    for dim_idx, dim in enumerate(dimensions):
        print(f"\n  Dimension d = {dim}")
        for std_idx, std in enumerate(noise_stds):
            m = means_noise[dim_idx, std_idx]
            s = stds_noise[dim_idx, std_idx]
            print(f"    std = {std:7.5f} -> mean angle = {m:8.4f}°, std = {s:8.4f}°")

    png_noise = f"{args.output_prefix}_noise_std.png"
    html_noise = f"{args.output_prefix}_noise_std.html"
    plot_noise_matplotlib(dimensions, noise_stds, means_noise, stds_noise, save_path=png_noise)
    print(f"\nSaved Experiment 1 PNG to: {png_noise}")
    try:
        plot_noise_html(dimensions, noise_stds, means_noise, stds_noise, html_path=html_noise)
        print(f"Saved Experiment 1 interactive HTML to: {html_noise}")
    except ModuleNotFoundError:
        print("Plotly is not installed; skipping Experiment 1 HTML export. Install with `pip install plotly`.")

    # -------- Experiment 2: alpha sweep --------
    means_alpha, stds_alpha = run_alpha_experiment(
        dimensions=dimensions,
        alphas=alphas,
        num_samples=args.num_samples,
        base_init_std=args.base_std,
        delta_init_std=args.delta_std,
        seed=args.seed + 1,  # different seed for variety
    )

    print("\nExperiment 2: Alpha sweep")
    for dim_idx, dim in enumerate(dimensions):
        print(f"\n  Dimension d = {dim}")
        for alpha_idx, alpha in enumerate(alphas):
            m = means_alpha[dim_idx, alpha_idx]
            s = stds_alpha[dim_idx, alpha_idx]
            print(f"    alpha = {alpha:5.2f} -> mean angle = {m:8.4f}°, std = {s:8.4f}°")

    png_alpha = f"{args.output_prefix}_alpha.png"
    html_alpha = f"{args.output_prefix}_alpha.html"
    plot_alpha_matplotlib(dimensions, alphas, means_alpha, stds_alpha, save_path=png_alpha)
    print(f"\nSaved Experiment 2 PNG to: {png_alpha}")
    try:
        plot_alpha_html(dimensions, alphas, means_alpha, stds_alpha, html_path=html_alpha)
        print(f"Saved Experiment 2 interactive HTML to: {html_alpha}")
    except ModuleNotFoundError:
        print("Plotly is not installed; skipping Experiment 2 HTML export. Install with `pip install plotly`.")


if __name__ == "__main__":
    main()

