import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
import plotly.graph_objects as go


def float_subset_values(exp_bits, mant_bits):
    """Generate all finite values for a custom floating point format.

    The values are mapped to numpy float16 so computations can rely on
    standard fp16 even though the representable set mimics lower-precision
    formats like e4m3 or e5m2.
    """
    bias = (1 << (exp_bits - 1)) - 1
    max_exp = (1 << exp_bits) - 1
    vals = []
    for sign in (0, 1):
        for exp in range(max_exp + 1):
            for mant in range(1 << mant_bits):
                if exp == max_exp:
                    # Skip inf/NaN representations
                    continue
                if exp == 0:
                    if mant == 0:
                        val = 0.0
                    else:
                        val = (-1)**sign * 2**(1 - bias) * (mant / (1 << mant_bits))
                else:
                    val = (-1)**sign * 2**(exp - bias) * (1 + mant / (1 << mant_bits))
                vals.append(np.float16(val))
    # ensure uniqueness and stable ordering
    return np.array(sorted(set(float(v) for v in vals)), dtype=np.float64)


def get_values(num_format, exp_bits=None, mant_bits=None):
    """Return representable numbers for the given format."""
    if num_format == "int3":
        return np.arange(-4, 4, dtype=np.float64)
    if num_format == "int4":
        return np.arange(-8, 8, dtype=np.float64)
    if num_format == "int5":
        return np.arange(-16, 16, dtype=np.float64)
    if num_format == "int6":
        return np.arange(-32, 32, dtype=np.float64)
    if num_format == "int7":
        return np.arange(-64, 64, dtype=np.float64)
    if num_format == "int8":
        return np.arange(-128, 128, dtype=np.float64)
    if num_format == "e4m3":
        exp_bits = exp_bits or 4
        mant_bits = mant_bits or 3
        return float_subset_values(exp_bits, mant_bits)
    if num_format == "e5m2":
        exp_bits = exp_bits or 5
        mant_bits = mant_bits or 2
        return float_subset_values(exp_bits, mant_bits)
    if num_format == "fp16":
        exp_bits = exp_bits or 5
        mant_bits = mant_bits or 10
        return float_subset_values(exp_bits, mant_bits)
    raise ValueError(f"unsupported format {num_format}")


def generate_vectors(values, mode, num_samples=None):
    """Generate 3D vectors either exhaustively or randomly."""
    if mode == "exhaustive":
        for combo in product(values, repeat=3):
            vec = np.array(combo, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                yield vec / norm
    else:
        assert num_samples is not None
        rnd = np.random.default_rng(0)
        for _ in range(num_samples):
            vec = rnd.choice(values, size=3)
            norm = np.linalg.norm(vec)
            if norm > 0:
                yield vec / norm


def bin_vectors(vectors, bins, projection="equal-area"):
    cos_thetas = []
    thetas = []
    phis = []
    for v in vectors:
        x, y, z = v
        r = np.linalg.norm(v)
        if r == 0:
            continue
        cos_theta = z / r
        phi = np.arctan2(y, x) % (2 * np.pi)
        phis.append(phi)
        cos_thetas.append(cos_theta)
        thetas.append(np.arccos(cos_theta))

    if projection == "equal-area":
        H, t_edges, p_edges = np.histogram2d(cos_thetas, phis, bins=bins,
                                             range=[[-1, 1], [0, 2 * np.pi]])
    else:
        H, t_edges, p_edges = np.histogram2d(thetas, phis, bins=bins,
                                             range=[[0, np.pi], [0, 2 * np.pi]])
    return H, t_edges, p_edges


def plot_heatmap(H, t_edges, p_edges, out_path, projection="equal-area", log_scale=False):
    plt.figure(figsize=(8, 4))
    if projection == "equal-area":
        img = plt.imshow(H, extent=[p_edges[0], p_edges[-1], t_edges[0], t_edges[-1]],
                         aspect='auto', origin='lower', cmap='hot',
                         norm=LogNorm() if log_scale else None)
        plt.ylabel('cos(theta)')
    else:
        img = plt.imshow(H, extent=[p_edges[0], p_edges[-1], t_edges[-1], t_edges[0]],
                         aspect='auto', cmap='hot',
                         norm=LogNorm() if log_scale else None)
        plt.ylabel('theta')
    plt.xlabel('phi')
    plt.colorbar(img, label='count' + (' (log)' if log_scale else ''))
    plt.title('Vector density on unit sphere')
    plt.tight_layout()
    plt.savefig(out_path)


def plot_heatmap_3d(H, t_edges, p_edges, out_path, projection="equal-area", log_scale=False):
    if projection == "equal-area":
        cos_centers = (t_edges[:-1] + t_edges[1:]) / 2
        theta_centers = np.arccos(cos_centers)
    else:
        theta_centers = (t_edges[:-1] + t_edges[1:]) / 2
    phi_centers = (p_edges[:-1] + p_edges[1:]) / 2
    Theta, Phi = np.meshgrid(theta_centers, phi_centers, indexing='ij')
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    data = np.log1p(H) if log_scale else H

    if out_path.lower().endswith('.html'):
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=data,
                                         colorscale='Hot', showscale=True)])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'))
        fig.write_html(out_path)
    else:
        norm = Normalize(vmin=data.min(), vmax=data.max())
        colors = cm.hot(norm(data))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1,
                        antialiased=False, shade=False)
        mappable = cm.ScalarMappable(cmap='hot', norm=norm)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.6,
                     label='count' + (' (log)' if log_scale else ''))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector distribution analysis")
    parser.add_argument('--format', choices=['int3', 'int4', 'int5', 'int6', 'int7', 'int8', 'e4m3', 'e5m2', 'fp16'], required=True)
    parser.add_argument('--mode', choices=['exhaustive', 'random'], default='exhaustive')
    parser.add_argument('--num', type=int, help='number of random samples for random mode')
    parser.add_argument('--bins', type=int, default=60, help='number of bins per dimension (rectangular tiling)')
    parser.add_argument('-e', '--exp', type=int, dest='exp_bits', help='number of exponent bits for floating formats')
    parser.add_argument('-m', '--mant', type=int, dest='mant_bits', help='number of mantissa bits for floating formats')
    parser.add_argument('--out', default='images/heatmap.png', help='output 2D heatmap path')
    parser.add_argument('--out3d', default=None,
                        help='optional 3D heatmap output path. Use a .html '
                             'extension for an interactive figure')
    parser.add_argument('--projection', choices=['equal-area', 'angular'],
                        default='equal-area',
                        help='heatmap projection / tiling scheme')
    parser.add_argument('--log', action='store_true', help='use log scale for heatmap')
    args = parser.parse_args()

    values = get_values(args.format, exp_bits=args.exp_bits, mant_bits=args.mant_bits)
    if args.mode == 'exhaustive':
        vectors = list(generate_vectors(values, 'exhaustive'))
    else:
        vectors = list(generate_vectors(values, 'random', num_samples=args.num))

    H, t_edges, p_edges = bin_vectors(vectors, bins=args.bins,
                                     projection=args.projection)
    plot_heatmap(H, t_edges, p_edges, args.out,
                 projection=args.projection, log_scale=args.log)
    print(f"Saved heatmap to {args.out}")
    if args.out3d:
        plot_heatmap_3d(H, t_edges, p_edges, args.out3d,
                        projection=args.projection, log_scale=args.log)
        print(f"Saved 3D heatmap to {args.out3d}")
