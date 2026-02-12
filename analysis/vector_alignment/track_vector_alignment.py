import argparse
import torch
import torch.nn.functional as F
import healpy as hp
import matplotlib.pyplot as plt


def int_quantize_ste(x: torch.Tensor, nbits: int) -> torch.Tensor:
    """Quantize x to the integer range determined by nbits using a straight-through estimator."""
    qmin = -(2 ** (nbits - 1))
    qmax = 2 ** (nbits - 1) - 1
    q = torch.clamp(torch.round(x), qmin, qmax)
    return x + (q - x).detach()


def normalize_ste(v: torch.Tensor, nbits: int) -> torch.Tensor:
    """Normalize v to unit length and quantize the result using the STE."""
    norm = torch.norm(v)
    normalized = v / norm.clamp(min=1e-8)
    return int_quantize_ste(normalized, nbits)


def plot_path(path, path_norm, nside):
    """Visualize the vector trajectory on a HEALPix sphere."""
    hp.mollview(
        torch.zeros(hp.nside2npix(nside)),
        title="3D Vector Alignment Trajectory",
        cbar=False,
        hold=True,
    )

    path_arr = torch.stack(path).numpy()
    theta, phi = hp.vec2ang(path_arr)
    hp.projplot(theta, phi, "k-", linewidth=1.0)
    hp.projscatter(theta, phi, c=range(len(path)), cmap="viridis", s=15)

    init_theta, init_phi = hp.vec2ang(path_arr[0])
    hp.projplot(init_theta, init_phi, "bo", markersize=5)

    cur_theta, cur_phi = hp.vec2ang(path_arr[-1])
    hp.projplot(cur_theta, cur_phi, "go", markersize=5)

    norm_arr = torch.stack(path_norm).numpy()
    n_theta, n_phi = hp.vec2ang(norm_arr[-1])
    hp.projplot(n_theta, n_phi, "mo", markersize=5)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Track the alignment of a learned 3D vector toward a target vector.")
    parser.add_argument("--nbits", type=int, default=4, help="Bit width for integer quantization (e.g. 4 for int4).")
    parser.add_argument("--init", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Initial x y z for the learned vector.")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 1.0, 0.0], help="Target x y z vector.")
    parser.add_argument("--steps", type=int, default=20, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--nside", type=int, default=8, help="HEALPix nside for visualization.")
    args = parser.parse_args()

    learned = torch.tensor(args.init, dtype=torch.float32, requires_grad=True)
    target = torch.tensor(args.target, dtype=torch.float32)
    optimizer = torch.optim.SGD([learned], lr=args.lr)

    path = [int_quantize_ste(learned, args.nbits).detach()]
    path_norm = [normalize_ste(learned, args.nbits).detach()]

    for _ in range(args.steps):
        optimizer.zero_grad()
        qvec = int_quantize_ste(learned, args.nbits)
        nvec = normalize_ste(qvec, args.nbits)
        loss = F.huber_loss(nvec, target)
        loss.backward()
        optimizer.step()
        path.append(int_quantize_ste(learned, args.nbits).detach())
        path_norm.append(normalize_ste(learned, args.nbits).detach())

    plot_path(path, path_norm, args.nside)


if __name__ == "__main__":
    main()
