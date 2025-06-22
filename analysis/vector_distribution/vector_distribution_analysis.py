import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def get_values(num_format):
    """Return representable numbers for the given format."""
    if num_format == "int3":
        return np.arange(-4, 4, dtype=np.float64)
    if num_format == "int4":
        return np.arange(-8, 8, dtype=np.float64)
    if num_format == "int8":
        return np.arange(-128, 128, dtype=np.float64)
    if num_format == "e4m3":
        # generate all fp8-e4m3 values
        vals = []
        for sign in [0, 1]:
            for exp in range(16):
                for mant in range(8):
                    bits = (sign << 7) | (exp << 3) | mant
                    # numpy 2.3 exposes float8 types
                    v = np.array(bits, dtype=np.float8_e4m3).astype(np.float32)
                    vals.append(float(v))
        return np.array(vals, dtype=np.float64)
    if num_format == "e5m2":
        vals = []
        for sign in [0, 1]:
            for exp in range(32):
                for mant in range(4):
                    bits = (sign << 7) | (exp << 2) | mant
                    v = np.array(bits, dtype=np.float8_e5m2).astype(np.float32)
                    vals.append(float(v))
        return np.array(vals, dtype=np.float64)
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


def bin_vectors(vectors, bins):
    thetas = []
    phis = []
    for v in vectors:
        x, y, z = v
        r = np.linalg.norm(v)
        if r == 0:
            continue
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x) % (2 * np.pi)
        thetas.append(theta)
        phis.append(phi)
    H, t_edges, p_edges = np.histogram2d(thetas, phis, bins=bins,
                                         range=[[0, np.pi], [0, 2 * np.pi]])
    return H, t_edges, p_edges


def plot_heatmap(H, t_edges, p_edges, out_path):
    plt.figure(figsize=(8, 4))
    plt.imshow(H, extent=[p_edges[0], p_edges[-1], t_edges[-1], t_edges[0]],
               aspect='auto', cmap='hot')
    plt.xlabel('phi')
    plt.ylabel('theta')
    plt.title('Vector density on unit sphere')
    plt.colorbar(label='count')
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector distribution analysis")
    parser.add_argument('--format', choices=['int3', 'int4', 'int8', 'e4m3', 'e5m2'], required=True)
    parser.add_argument('--mode', choices=['exhaustive', 'random'], default='exhaustive')
    parser.add_argument('--num', type=int, help='number of random samples for random mode')
    parser.add_argument('--bins', type=int, default=60, help='number of bins per dimension')
    parser.add_argument('--out', default='images/heatmap.png', help='output image path')
    args = parser.parse_args()

    values = get_values(args.format)
    if args.mode == 'exhaustive':
        vectors = list(generate_vectors(values, 'exhaustive'))
    else:
        vectors = list(generate_vectors(values, 'random', num_samples=args.num))

    H, t_edges, p_edges = bin_vectors(vectors, bins=args.bins)
    plot_heatmap(H, t_edges, p_edges, args.out)
    print(f"Saved heatmap to {args.out}")
