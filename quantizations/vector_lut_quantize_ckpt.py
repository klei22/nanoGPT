#!/usr/bin/env python3
"""Vector lookup-table quantization for nanoGPT checkpoints.

This script replaces floating-point tensors with the closest vector from a
pre-generated spherical codebook. Each vector in the LUT is compared against the
rows of eligible tensors (default: last dimension equals the embedding size),
and the row is projected onto the best-matching LUT direction.  Unlike uniform
fake quantization, this method preserves directions from an explicit dictionary
of vectors sampled via low-discrepancy hypersphere grids or lattices.

Examples
--------
python quantizations/vector_lut_quantize_ckpt.py out_ptq_demo \
    --out_dir out_ptq_demo_lut --lut-size 100000 --method grid_kronecker

python quantizations/vector_lut_quantize_ckpt.py out_fake_ptq_minipile \
    --out_dir out_fake_ptq_minipile_lut --lut-size 10000 \
    --method gaussian_baseline --gaussian-std 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


# ----------------------------- LUT generation ---------------------------------


def _unit_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _prime_sieve(n: int) -> np.ndarray:
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i : n + 1 : i] = False
    return np.flatnonzero(sieve)


def _first_primes(k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    if k < 6:
        bound = 15
    else:
        bound = int(k * (math.log(k) + math.log(math.log(k))) * 1.2) + 10
    primes = _prime_sieve(bound)
    while len(primes) < k:
        bound *= 2
        primes = _prime_sieve(bound)
    return primes[:k]


def _kronecker_sequence(dim: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shift = rng.uniform(0.0, 1.0)
    primes = _first_primes(dim)
    alphas = np.sqrt(primes.astype(np.float64) + shift)
    frac = np.modf(alphas)[0]
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    offset = np.array([math.modf(phi ** (i + 1))[0] for i in range(dim)], dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)[:, None]
    seq = (k * frac[None, :] + offset[None, :]) % 1.0
    return seq


def _halton_sequence(dim: int, n: int, start_index: int = 1) -> np.ndarray:
    bases = _first_primes(dim)
    seq = np.empty((n, dim), dtype=np.float64)
    for j, b in enumerate(bases):
        seq[:, j] = np.array([
            _radical_inverse(i, int(b)) for i in range(start_index, start_index + n)
        ])
    return seq


def _radical_inverse(i: int, base: int) -> float:
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _u01_to_gaussian(u01: np.ndarray) -> np.ndarray:
    # Vectorized inverse CDF via Acklam approximation (borrowed conceptually from
    # analysis/hypersphere_grid, but reimplemented here for minimal dependencies).
    p = np.asarray(u01, dtype=np.float64)
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )
    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1.0 - eps)
    pl = p < 0.02425
    pu = p > 1.0 - 0.02425
    pm = ~(pl | pu)
    x = np.empty_like(p, dtype=np.float64)
    if np.any(pl):
        q = np.sqrt(-2.0 * np.log(p[pl]))
        x[pl] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(pm):
        q = p[pm] - 0.5
        r = q * q
        x[pm] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    if np.any(pu):
        q = np.sqrt(-2.0 * np.log(1.0 - p[pu]))
        x[pu] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    return x


def _sequence_to_sphere(u01: np.ndarray) -> np.ndarray:
    z = _u01_to_gaussian(u01)
    return _unit_norm(z, axis=1)


def _generate_lut(method: str, n: int, dim: int, *, seed: int, gaussian_std: float) -> np.ndarray:
    method = method.lower()
    if n <= 0:
        raise ValueError("--lut-size must be positive")
    if dim <= 0:
        raise ValueError("--vector-dim must be positive")

    if method in {"grid_kronecker", "grid:kronecker"}:
        u = _kronecker_sequence(dim, n, seed=seed)
        lut = _sequence_to_sphere(u)
    elif method in {"grid_halton", "grid:halton"}:
        u = _halton_sequence(dim, n, start_index=1)
        lut = _sequence_to_sphere(u)
    elif method in {"grid_random", "grid:random"}:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(size=(n, dim))
        lut = _unit_norm(z, axis=1)
    elif method in {"lattice_rseq", "lattice:rseq"}:
        primes = _first_primes(dim)
        alpha = np.sqrt(primes).astype(np.float64)
        alpha = np.modf(alpha)[0]
        k = np.arange(n, dtype=np.float64)[:, None]
        offset = 0.5
        u = (offset + k * alpha[None, :]) % 1.0
        lut = _sequence_to_sphere(u)
    elif method in {"lattice_halton", "lattice:halton"}:
        u = _halton_sequence(max(dim, 2), n, start_index=1)
        lut = _sequence_to_sphere(u[:, :dim])
    elif method in {"lattice_random", "lattice:random"}:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(size=(n, dim))
        lut = _unit_norm(z, axis=1)
    elif method in {"gaussian_baseline", "baseline_gaussian"}:
        rng = np.random.default_rng(seed)
        z = rng.normal(loc=0.0, scale=gaussian_std, size=(n, dim))
        lut = z.astype(np.float64, copy=False)
    else:
        raise ValueError(
            "Unknown LUT method '{method}'. Expected grid_kronecker, grid_halton, "
            "grid_random, lattice_rseq, lattice_halton, lattice_random, or gaussian_baseline.".format(
                method=method
            )
        )
    return lut.astype(np.float32, copy=False)


# --------------------------- Quantization logic --------------------------------


def _chunked_argmax_dot(
    rows: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return indices and signed dot products of best matches for each row."""
    if rows.numel() == 0:
        return (
            torch.empty(rows.shape[0], dtype=torch.long, device=rows.device),
            torch.empty(rows.shape[0], dtype=rows.dtype, device=rows.device),
        )

    best_abs = torch.full((rows.shape[0],), -float("inf"), dtype=rows.dtype, device=rows.device)
    best_scores = torch.zeros((rows.shape[0],), dtype=rows.dtype, device=rows.device)
    best_indices = torch.zeros((rows.shape[0],), dtype=torch.long, device=rows.device)

    for start in range(0, codebook.shape[0], chunk_size):
        end = min(start + chunk_size, codebook.shape[0])
        chunk = codebook[start:end]
        scores = rows @ chunk.t()
        abs_scores = scores.abs()
        vals, idx = abs_scores.max(dim=1)
        improved = vals > best_abs
        if torch.any(improved):
            best_abs[improved] = vals[improved]
            gathered = scores[improved, :]
            idx_improved = idx[improved]
            row_idx = torch.arange(
                idx_improved.numel(), device=rows.device, dtype=torch.long
            )
            best_scores[improved] = gathered[row_idx, idx_improved]
            best_indices[improved] = idx_improved.to(torch.long) + start
    return best_indices, best_scores


def _quantize_tensor(
    tensor: torch.Tensor,
    *,
    vector_dim: int,
    codebook: torch.Tensor,
    codebook_norm_sq: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return tensor
    if tensor.numel() == 0:
        return tensor
    if tensor.shape[-1] != vector_dim:
        return tensor

    orig_dtype = tensor.dtype
    rows = tensor.reshape(-1, vector_dim).to(torch.float32)
    indices, scores = _chunked_argmax_dot(rows, codebook, chunk_size=chunk_size)
    if indices.numel() == 0:
        return tensor
    selected = codebook.index_select(0, indices)
    denom = codebook_norm_sq.index_select(0, indices)
    denom = torch.clamp(denom, min=1e-12)
    scales = scores / denom
    quantized_rows = selected * scales.unsqueeze(1)
    return quantized_rows.reshape(tensor.shape).to(orig_dtype)


# ----------------------------- CLI + I/O ---------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply LUT-based vector quantization to a checkpoint.")
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and meta.pkl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to store the quantized checkpoint (defaults to <ckpt_dir>_lut)",
    )
    parser.add_argument(
        "--lut-size",
        type=int,
        default=10000,
        help="Number of vectors in the lookup table",
    )
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=384,
        help="Target vector dimensionality (default: 384)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid_kronecker",
        help="LUT generation method (grid_kronecker, grid_halton, grid_random, "
        "lattice_rseq, lattice_halton, lattice_random, gaussian_baseline)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=65536,
        help="Number of codebook vectors to process per batch when searching",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stochastic LUT generation",
    )
    parser.add_argument(
        "--gaussian-std",
        type=float,
        default=0.02,
        help="Standard deviation for gaussian_baseline method",
    )
    parser.add_argument(
        "--augment-negatives",
        action="store_true",
        help="Augment the LUT with negative copies of each vector (doubles LUT size)",
    )
    parser.add_argument(
        "--skip-copy-meta",
        action="store_true",
        help="Do not copy auxiliary files (meta.pkl, config) to out_dir",
    )
    return parser.parse_args()


@dataclass
class QuantizationSummary:
    tensors_total: int
    tensors_quantized: int
    skipped_shape_mismatch: int
    skipped_dtype: int
    vector_dim: int
    lut_size: int
    method: str

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def main() -> None:
    args = parse_args()
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.abspath(f"{ckpt_dir}_lut")
    )
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_obj = checkpoint["model"]
    else:
        state_obj = checkpoint

    if isinstance(state_obj, dict):
        state_dict = state_obj
    else:
        to_state_dict = getattr(state_obj, "state_dict", None)
        if callable(to_state_dict):
            state_dict = to_state_dict()
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint["model"] = state_dict
            else:
                checkpoint = state_dict
        else:
            raise TypeError("Unsupported checkpoint format: expected a mapping")

    lut_np = _generate_lut(
        args.method,
        args.lut_size,
        args.vector_dim,
        seed=args.seed,
        gaussian_std=args.gaussian_std,
    )
    if args.augment_negatives:
        lut_np = np.concatenate([lut_np, -lut_np], axis=0)
    codebook = torch.from_numpy(lut_np.astype(np.float32, copy=False))
    codebook_norm_sq = (codebook * codebook).sum(dim=1)

    total = 0
    quantized = 0
    skipped_shape = 0
    skipped_dtype = 0

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        total += 1
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            skipped_dtype += 1
            continue
        if tensor.shape[-1] != args.vector_dim:
            skipped_shape += 1
            continue
        quantized_tensor = _quantize_tensor(
            tensor,
            vector_dim=args.vector_dim,
            codebook=codebook,
            codebook_norm_sq=codebook_norm_sq,
            chunk_size=max(1, args.chunk_size),
        )
        state_dict[name] = quantized_tensor
        quantized += 1

    quant_summary = QuantizationSummary(
        tensors_total=total,
        tensors_quantized=quantized,
        skipped_shape_mismatch=skipped_shape,
        skipped_dtype=skipped_dtype,
        vector_dim=args.vector_dim,
        lut_size=codebook.shape[0],
        method=args.method,
    )

    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    config = {
        "ckpt_dir": ckpt_dir,
        "out_dir": out_dir,
        "vector_dim": args.vector_dim,
        "lut_size": int(codebook.shape[0]),
        "method": args.method,
        "seed": args.seed,
        "gaussian_std": args.gaussian_std,
        "augment_negatives": bool(args.augment_negatives),
        "chunk_size": int(args.chunk_size),
        "summary": quant_summary.to_dict(),
    }
    with open(os.path.join(out_dir, "lut_quant_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    np.save(os.path.join(out_dir, "lut_vectors.npy"), codebook.numpy())

    if not args.skip_copy_meta:
        meta_src = os.path.join(ckpt_dir, "meta.pkl")
        if os.path.exists(meta_src):
            shutil.copy2(meta_src, os.path.join(out_dir, "meta.pkl"))

    print("Vector LUT quantization complete.")
    print(json.dumps(config["summary"], indent=2))


if __name__ == "__main__":
    main()
