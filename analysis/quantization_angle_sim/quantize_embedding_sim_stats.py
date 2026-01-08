#!/usr/bin/env python3
"""
Quantize random embedding vectors and analyze distortion vs the original.

Features
--------
- Two fake quantizers:
  1) Integer symmetric per-vector scaling (int8 -> int3)
  2) Custom floating e/m (e exponent bits, m mantissa bits), e.g., e4m3
- Backends: NumPy or PyTorch (select via --framework)
- Set standard deviation (--std) and dtype (--dtype)
- Reproducible RNG (seed applies to the actual generator used)
- Chained mode (--chain): progressively quantize the previous dequantized output
  but always measure against the original float vectors.

Metrics per quantization level
------------------------------
- Cosine similarity
- Angle delta (degrees)
- Magnitude ratio = ||x_hat|| / ||x||
- Magnitude delta (absolute, %) = | ||x_hat|| / ||x|| - 1 | * 100

Outputs
-------
- CSV per D: {int|em}_stats_D{D}.csv with columns:
  label, bits (None for EM), [e_bits, m_bits if EM],
  cos_mean, cos_std, deg_mean, deg_std,
  mag_ratio_mean, mag_ratio_std,
  mag_delta_pct_mean, mag_delta_pct_std,
  mag_delta_pct_signed_mean, mag_delta_pct_signed_std
- Plots per D:
  {int|em}_angle_deg_D{D}.png,
  {int|em}_cosine_D{D}.png,
  {int|em}_mag_delta_D{D}.png              # NEW
- Combined plots across all D:
  {int|em}_angle_deg_ALL.png,
  {int|em}_cosine_ALL.png,
  {int|em}_mag_delta_ALL.png               # NEW

Usage examples
--------------
# INT (default quantizer), bits 8..3, Torch backend, float32
python quantize_embedding_sim_stats.py --quantizer int --bits-from 8 --bits-to 3

# e/m single format (e4m3), NumPy backend, float64
python quantize_embedding_sim_stats.py --framework numpy --dtype float64 --quantizer em -e 4 -m 3

# e/m sweep
python quantize_embedding_sim_stats.py --quantizer em --em-list "4,3 5,2 5,3 6,2"

# Torch + bfloat16 + GPU
python quantize_embedding_sim_stats.py --framework torch --device cuda --dtype bfloat16 --quantizer em -e 4 -m 3
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional torch import (only needed if --framework torch)
try:
    import torch
except Exception:
    torch = None


# ---------------------- DTYPE HELPERS ----------------------
NP_ALLOWED_DTYPES = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
TORCH_ALLOWED_DTYPES = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "bfloat16": "bfloat16",
}

def parse_numpy_dtype(name: str):
    if name not in NP_ALLOWED_DTYPES:
        raise ValueError(f"Unsupported numpy dtype '{name}'. Choose from {list(NP_ALLOWED_DTYPES.keys())}.")
    return NP_ALLOWED_DTYPES[name]

def parse_torch_dtype(name: str):
    if torch is None:
        raise RuntimeError("PyTorch not available but --framework torch was requested.")
    if name not in TORCH_ALLOWED_DTYPES:
        raise ValueError(f"Unsupported torch dtype '{name}'. Choose from {list(TORCH_ALLOWED_DTYPES.keys())}.")
    return getattr(torch, TORCH_ALLOWED_DTYPES[name])


# ---------------------- METRICS (NUMPY & TORCH) ----------------------
def np_metrics(x: np.ndarray, y: np.ndarray):
    """Return (cos, angle_deg, mag_ratio, mag_delta_pct_abs, mag_delta_pct_signed) per row."""
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)

    num = np.einsum("ij,ij->i", x64, y64)
    nx = np.linalg.norm(x64, axis=1)
    ny = np.linalg.norm(y64, axis=1)

    # Cosine & angle
    denom = nx * ny
    cos = num / np.maximum(denom, 1e-12)
    cos = np.clip(cos, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))

    # Magnitude ratio & deltas
    mag_ratio = ny / np.maximum(nx, 1e-12)
    mag_delta_pct_signed = (mag_ratio - 1.0) * 100.0
    mag_delta_pct_abs = np.abs(mag_delta_pct_signed)

    return cos, angle_deg, mag_ratio, mag_delta_pct_abs, mag_delta_pct_signed

def th_metrics(x: "torch.Tensor", y: "torch.Tensor"):
    """Return (cos, angle_deg, mag_ratio, mag_delta_pct_abs, mag_delta_pct_signed) per row."""
    # Upcast for stability if needed
    xx = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    yy = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y

    num = (xx * yy).sum(dim=1)
    nx = torch.linalg.norm(xx, dim=1)
    ny = torch.linalg.norm(yy, dim=1)

    # Cosine & angle
    denom = nx * ny
    eps = torch.tensor(1e-12, dtype=denom.dtype, device=denom.device)
    cos = num / torch.maximum(denom, eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    angle = torch.rad2deg(torch.arccos(cos))

    # Magnitude ratio & deltas
    mag_ratio = ny / torch.maximum(nx, eps)
    mag_delta_pct_signed = (mag_ratio - 1.0) * 100.0
    mag_delta_pct_abs = torch.abs(mag_delta_pct_signed)

    # Return as numpy arrays
    return (
        cos.detach().cpu().numpy(),
        angle.detach().cpu().numpy(),
        mag_ratio.detach().cpu().numpy(),
        mag_delta_pct_abs.detach().cpu().numpy(),
        mag_delta_pct_signed.detach().cpu().numpy(),
    )


# ---------------------- NUMPY BACKEND ----------------------
def np_draw_vectors(n: int, d: int, dist: str, std: float, dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "normal":
        x = rng.standard_normal((n, d)).astype(np.float64) * std
    elif dist == "uniform":
        x = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float64)
        x *= std / (np.sqrt(1.0 / 3.0) + 1e-12)  # scale to requested std
    else:
        raise ValueError("dist must be either 'normal' or 'uniform'")
    return x.astype(dtype, copy=False)

def np_quantize_int_per_vector(x: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1:
        raise ValueError("bits must be >= 1")
    qmax = (1 << (bits - 1)) - 1
    max_abs = np.max(np.abs(x), axis=1)  # (N,)
    tiny = np.finfo(x.dtype).tiny if np.issubdtype(x.dtype, np.floating) else 1e-12
    scales = np.maximum(max_abs / qmax, tiny)
    q = np.round(x / scales[:, None])
    q = np.clip(q, -qmax, qmax)
    return (q * scales[:, None]).astype(x.dtype, copy=False)

def _np_em_edges(e_bits: int, m_bits: int):
    # Bias for exponent, reserve all-ones exponent for inf/NaN (we saturate to max finite)
    bias = (1 << (e_bits - 1)) - 1
    emax = (1 << e_bits) - 2 - bias      # max unbiased exponent for normalized (all-ones reserved)
    emin = 1 - bias                       # min unbiased exponent for normalized
    max_norm = (2.0 - 2.0**(-m_bits)) * (2.0 ** emax)
    min_norm = 2.0 ** emin                # smallest normalized positive
    sub_step = 2.0 ** (emin - m_bits)     # subnormal quantum (min subnormal = sub_step)
    return bias, emax, emin, max_norm, min_norm, sub_step

def np_quantize_em(x: np.ndarray, e_bits: int, m_bits: int) -> np.ndarray:
    """
    Round to nearest representable value in a custom e/m floating format:
      - 1 sign bit, e_bits exponent bits (bias = 2^{e_bits-1}-1), m_bits mantissa
      - all-ones exponent treated as inf/NaN; we saturate to max finite
      - subnormals supported via step = 2^{emin - m_bits}
    """
    if e_bits < 2 or m_bits < 1:
        raise ValueError("e_bits >= 2 and m_bits >= 1 are required.")

    ax = np.abs(x).astype(np.float64)
    sign = np.sign(x).astype(np.float64)
    out = np.empty_like(ax)

    bias, emax, emin, max_norm, min_norm, sub_step = _np_em_edges(e_bits, m_bits)

    # Zeros
    mask_zero = (ax == 0)
    out[mask_zero] = 0.0

    # Overflow -> saturate to max finite
    mask_over = ax >= max_norm
    out[mask_over] = max_norm

    # Subnormals: 0 < ax < min_norm
    mask_sub = (ax > 0) & (ax < min_norm)
    if np.any(mask_sub):
        q = np.round(ax[mask_sub] / sub_step)
        q = np.clip(q, 0, (1 << m_bits) - 1)
        out[mask_sub] = q * sub_step

    # Normal range
    mask_norm = (ax >= min_norm) & (ax < max_norm)
    if np.any(mask_norm):
        m, e = np.frexp(ax[mask_norm])  # ax = m * 2^e, m in [0.5, 1)
        frac = (m * 2.0) - 1.0          # in [0, 1)
        frac_q = np.round(frac * (1 << m_bits)) / (1 << m_bits)

        # Handle rounding carry into next exponent
        carry = frac_q >= 1.0
        frac_q[carry] = 0.0
        e = e + carry.astype(int)
        E = e - 1                         # unbiased exponent for normalized

        # If carry overflows exponent, saturate to max finite
        overflow = E > emax
        if np.any(overflow):
            E[overflow] = emax
            frac_q[overflow] = 1.0 - 2.0**(-m_bits)  # S = 2 - 2^-m at Emax

        S = 1.0 + frac_q
        out_norm = S * (2.0 ** E)

        # Rare case: if rounding pushed us below min_norm, quantize as subnormal
        below = out_norm < min_norm
        if np.any(below):
            q = np.round(out_norm[below] / sub_step)
            q = np.clip(q, 0, (1 << m_bits) - 1)
            out_norm[below] = q * sub_step

        out[mask_norm] = out_norm

    return (out * sign).astype(x.dtype, copy=False)

def np_run_trial_int(n_vectors: int, emb_dim: int, bits_list: List[int], seed: int,
                     dist: str, chain: bool, std: float, dtype) -> pd.DataFrame:
    x = np_draw_vectors(n_vectors, emb_dim, dist, std, dtype, seed)
    results = []
    base = x
    for bits in bits_list:
        xhat = np_quantize_int_per_vector(base if chain else x, bits)
        cos, angle, mag_ratio, mag_abs_pct, mag_signed_pct = np_metrics(x, xhat)
        results.append(dict(
            bits=bits,
            label=f"int{bits}",
            cos_mean=float(np.mean(cos)),
            cos_std=float(np.std(cos, ddof=1)),
            deg_mean=float(np.mean(angle)),
            deg_std=float(np.std(angle, ddof=1)),
            mag_ratio_mean=float(np.mean(mag_ratio)),
            mag_ratio_std=float(np.std(mag_ratio, ddof=1)),
            mag_delta_pct_mean=float(np.mean(mag_abs_pct)),
            mag_delta_pct_std=float(np.std(mag_abs_pct, ddof=1)),
            mag_delta_pct_signed_mean=float(np.mean(mag_signed_pct)),
            mag_delta_pct_signed_std=float(np.std(mag_signed_pct, ddof=1)),
        ))
        if chain:
            base = xhat
    # Keep numeric order (descending bits) as provided by bits_list
    df = pd.DataFrame(results)
    order = [f"int{b}" for b in bits_list]
    return df.set_index("label").loc[order].reset_index()

def np_run_trial_em(n_vectors: int, emb_dim: int, em_list: List[Tuple[int, int]], seed: int,
                    dist: str, chain: bool, std: float, dtype) -> pd.DataFrame:
    x = np_draw_vectors(n_vectors, emb_dim, dist, std, dtype, seed)
    results = []
    base = x
    for e_bits, m_bits in em_list:
        xhat = np_quantize_em(base if chain else x, e_bits, m_bits)
        cos, angle, mag_ratio, mag_abs_pct, mag_signed_pct = np_metrics(x, xhat)
        results.append(dict(
            bits=None,
            label=f"e{e_bits}m{m_bits}",
            e_bits=e_bits,
            m_bits=m_bits,
            cos_mean=float(np.mean(cos)),
            cos_std=float(np.std(cos, ddof=1)),
            deg_mean=float(np.mean(angle)),
            deg_std=float(np.std(angle, ddof=1)),
            mag_ratio_mean=float(np.mean(mag_ratio)),
            mag_ratio_std=float(np.std(mag_ratio, ddof=1)),
            mag_delta_pct_mean=float(np.mean(mag_abs_pct)),
            mag_delta_pct_std=float(np.std(mag_abs_pct, ddof=1)),
            mag_delta_pct_signed_mean=float(np.mean(mag_signed_pct)),
            mag_delta_pct_signed_std=float(np.std(mag_signed_pct, ddof=1)),
        ))
        if chain:
            base = xhat
    # Preserve the provided em_list ordering
    order = [f"e{e}m{m}" for e, m in em_list]
    df = pd.DataFrame(results)
    return df.set_index("label").loc[order].reset_index()


# ---------------------- TORCH BACKEND ----------------------
def th_draw_vectors(n: int, d: int, dist: str, std: float, dtype, device: str, seed: int) -> "torch.Tensor":
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if dist == "normal":
        x = torch.randn((n, d), dtype=torch.float64, device=device, generator=g) * std
    elif dist == "uniform":
        x = (torch.rand((n, d), dtype=torch.float64, device=device, generator=g) * 2.0 - 1.0)
        x *= std / (np.sqrt(1.0 / 3.0) + 1e-12)
    else:
        raise ValueError("dist must be either 'normal' or 'uniform'")
    return x.to(dtype)

def th_quantize_int_per_vector(x: "torch.Tensor", bits: int) -> "torch.Tensor":
    if bits < 1:
        raise ValueError("bits must be >= 1")
    qmax = (1 << (bits - 1)) - 1
    max_abs = torch.amax(torch.abs(x), dim=1)
    tiny = torch.tensor(torch.finfo(x.dtype).tiny, dtype=x.dtype, device=x.device)
    scales = torch.maximum(max_abs / qmax, tiny)
    q = torch.round(x / scales[:, None])
    q = torch.clamp(q, -qmax, qmax)
    return (q * scales[:, None]).to(dtype=x.dtype)

def _th_em_edges(e_bits: int, m_bits: int, device, dtype=torch.float64):
    bias = (1 << (e_bits - 1)) - 1
    emax = (1 << e_bits) - 2 - bias
    emin = 1 - bias
    two = torch.tensor(2.0, device=device, dtype=dtype)
    max_norm = (two - two.pow(-m_bits)) * two.pow(emax)
    min_norm = two.pow(emin)
    sub_step = two.pow(emin - m_bits)
    return bias, emax, emin, max_norm, min_norm, sub_step

def th_quantize_em(x: "torch.Tensor", e_bits: int, m_bits: int) -> "torch.Tensor":
    if e_bits < 2 or m_bits < 1:
        raise ValueError("e_bits >= 2 and m_bits >= 1 are required.")
    device = x.device
    dtype64 = torch.float64

    ax = torch.abs(x).to(dtype64)
    sign = torch.sign(x).to(dtype64)
    out = torch.empty_like(ax)

    bias, emax, emin, max_norm, min_norm, sub_step = _th_em_edges(e_bits, m_bits, device, dtype64)

    # zeros
    mask_zero = ax == 0
    out[mask_zero] = 0.0

    # overflow -> saturate to max finite
    mask_over = ax >= max_norm
    out[mask_over] = max_norm

    # subnormals
    mask_sub = (ax > 0) & (ax < min_norm)
    if mask_sub.any():
        q = torch.round(ax[mask_sub] / sub_step)
        q = torch.clamp(q, 0, (1 << m_bits) - 1)
        out[mask_sub] = q * sub_step

    # normals
    mask_norm = (ax >= min_norm) & (ax < max_norm)
    if mask_norm.any():
        m, e = torch.frexp(ax[mask_norm])  # m in [0.5, 1)
        frac = (m * 2.0) - 1.0
        frac_q = torch.round(frac * (1 << m_bits)) / (1 << m_bits)
        carry = frac_q >= 1.0
        frac_q = torch.where(carry, torch.tensor(0.0, device=device, dtype=dtype64), frac_q)
        e = e + carry.to(e.dtype)
        E = e - 1.0

        # overflow from carry: saturate to max finite
        overflow = E > float(emax)
        if overflow.any():
            E = torch.where(overflow, torch.tensor(float(emax), device=device, dtype=E.dtype), E)
            frac_q = torch.where(
                overflow,
                torch.tensor(1.0 - 2.0**(-m_bits), device=device, dtype=dtype64),
                frac_q,
            )

        S = 1.0 + frac_q
        out_norm = S * torch.pow(torch.tensor(2.0, device=device, dtype=dtype64), E)

        # rare: if below min_norm after rounding, quantize as subnormal
        below = out_norm < min_norm
        if below.any():
            q = torch.round(out_norm[below] / sub_step)
            q = torch.clamp(q, 0, (1 << m_bits) - 1)
            out_norm[below] = q * sub_step

        out[mask_norm] = out_norm

    return (out * sign).to(dtype=x.dtype)

def th_run_trial_int(n_vectors: int, emb_dim: int, bits_list: List[int], seed: int,
                     dist: str, chain: bool, std: float, dtype, device: str) -> pd.DataFrame:
    x = th_draw_vectors(n_vectors, emb_dim, dist, std, dtype, device, seed)
    results = []
    base = x
    for bits in bits_list:
        xhat = th_quantize_int_per_vector(base if chain else x, bits)
        cos, angle, mag_ratio, mag_abs_pct, mag_signed_pct = th_metrics(x, xhat)
        results.append(dict(
            bits=bits,
            label=f"int{bits}",
            cos_mean=float(np.mean(cos)),
            cos_std=float(np.std(cos, ddof=1)),
            deg_mean=float(np.mean(angle)),
            deg_std=float(np.std(angle, ddof=1)),
            mag_ratio_mean=float(np.mean(mag_ratio)),
            mag_ratio_std=float(np.std(mag_ratio, ddof=1)),
            mag_delta_pct_mean=float(np.mean(mag_abs_pct)),
            mag_delta_pct_std=float(np.std(mag_abs_pct, ddof=1)),
            mag_delta_pct_signed_mean=float(np.mean(mag_signed_pct)),
            mag_delta_pct_signed_std=float(np.std(mag_signed_pct, ddof=1)),
        ))
        if chain:
            base = xhat
    df = pd.DataFrame(results)
    order = [f"int{b}" for b in bits_list]
    return df.set_index("label").loc[order].reset_index()

def th_run_trial_em(n_vectors: int, emb_dim: int, em_list: List[Tuple[int, int]], seed: int,
                    dist: str, chain: bool, std: float, dtype, device: str) -> pd.DataFrame:
    x = th_draw_vectors(n_vectors, emb_dim, dist, std, dtype, device, seed)
    results = []
    base = x
    for e_bits, m_bits in em_list:
        xhat = th_quantize_em(base if chain else x, e_bits, m_bits)
        cos, angle, mag_ratio, mag_abs_pct, mag_signed_pct = th_metrics(x, xhat)
        results.append(dict(
            bits=None,
            label=f"e{e_bits}m{m_bits}",
            e_bits=e_bits,
            m_bits=m_bits,
            cos_mean=float(np.mean(cos)),
            cos_std=float(np.std(cos, ddof=1)),
            deg_mean=float(np.mean(angle)),
            deg_std=float(np.std(angle, ddof=1)),
            mag_ratio_mean=float(np.mean(mag_ratio)),
            mag_ratio_std=float(np.std(mag_ratio, ddof=1)),
            mag_delta_pct_mean=float(np.mean(mag_abs_pct)),
            mag_delta_pct_std=float(np.std(mag_abs_pct, ddof=1)),
            mag_delta_pct_signed_mean=float(np.mean(mag_signed_pct)),
            mag_delta_pct_signed_std=float(np.std(mag_signed_pct, ddof=1)),
        ))
        if chain:
            base = xhat
    order = [f"e{e}m{m}" for e, m in em_list]
    df = pd.DataFrame(results)
    return df.set_index("label").loc[order].reset_index()


# ---------------------- PLOTTING ----------------------
def plot_per_dim(df: pd.DataFrame, emb_dim: int, outdir: str, quantizer: str) -> None:
    """Three plots per D: angle delta, cosine similarity, magnitude delta (%), each with std error bars."""
    os.makedirs(outdir, exist_ok=True)
    labels = df["label"].tolist()
    x = list(range(len(labels)))

    # Angle
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, df["deg_mean"], yerr=df["deg_std"], marker="o", capsize=4, linewidth=2)
    plt.xticks(x, labels)
    plt.xlabel("Quantization level")
    plt.ylabel("Angle delta (degrees)")
    plt.title(f"Angle vs. Quantization (D={emb_dim})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{quantizer}_angle_deg_D{emb_dim}.png"), dpi=150)
    plt.close()

    # Cosine
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, df["cos_mean"], yerr=df["cos_std"], marker="o", capsize=4, linewidth=2)
    plt.xticks(x, labels)
    plt.xlabel("Quantization level")
    plt.ylabel("Cosine similarity")
    plt.title(f"Cosine Similarity vs. Quantization (D={emb_dim})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{quantizer}_cosine_D{emb_dim}.png"), dpi=150)
    plt.close()

    # Magnitude delta (absolute, %)
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, df["mag_delta_pct_mean"], yerr=df["mag_delta_pct_std"], marker="o", capsize=4, linewidth=2)
    plt.xticks(x, labels)
    plt.xlabel("Quantization level")
    plt.ylabel("Magnitude delta (|Δ|, %)")
    plt.title(f"Magnitude Delta vs. Quantization (D={emb_dim})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{quantizer}_mag_delta_D{emb_dim}.png"), dpi=150)
    plt.close()

def plot_combined(dfs_by_dim: Dict[int, pd.DataFrame], labels: List[str], outdir: str, quantizer: str) -> None:
    """Overlay plots across all embedding sizes, aligned to the given label order."""
    os.makedirs(outdir, exist_ok=True)
    x = list(range(len(labels)))

    # Angle
    plt.figure(figsize=(10, 6))
    for d, df in dfs_by_dim.items():
        dd = df.set_index("label").loc[labels].reset_index()
        plt.errorbar(x, dd["deg_mean"], yerr=dd["deg_std"], marker="o", capsize=3, linewidth=2, label=f"D={d}")
    plt.xticks(x, labels)
    plt.xlabel("Quantization level")
    plt.ylabel("Angle delta (degrees)")
    plt.title(f"Angle vs. Quantization (All Embedding Sizes) [{quantizer}]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{quantizer}_angle_deg_ALL.png"), dpi=150)
    plt.close()

    # Cosine
    plt.figure(figsize=(10, 6))
    for d, df in dfs_by_dim.items():
        dd = df.set_index("label").loc[labels].reset_index()
        plt.errorbar(x, dd["cos_mean"], yerr=dd["cos_std"], marker="o", capsize=3, linewidth=2, label=f"D={d}")
    plt.xticks(x, labels)
    plt.xlabel("Quantization level")
    plt.ylabel("Cosine similarity")
    plt.title(f"Cosine Similarity vs. Quantization (All Embedding Sizes) [{quantizer}]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{quantizer}_cosine_ALL.png"), dpi=150)
    plt.close()

    # Magnitude delta (absolute, %)
    plt.figure(figsize=(10, 6))
    for d, df in dfs_by_dim.items():
        dd = df.set_index("label").loc[labels].reset_index()
        plt.errorbar(x, dd["mag_delta_pct_mean"], yerr=dd["mag_delta_pct_std"], marker="o", capsize=3, linewidth=2, label=f"D={d}")
    plt.xticks(x, labels)
    plt.xlabel("Quantization level")
    plt.ylabel("Magnitude delta (|Δ|, %)")
    plt.title(f"Magnitude Delta vs. Quantization (All Embedding Sizes) [{quantizer}]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{quantizer}_mag_delta_ALL.png"), dpi=150)
    plt.close()


# ---------------------- MAIN ----------------------
def parse_em_list(em_list_str: str) -> List[Tuple[int, int]]:
    pairs = []
    for token in em_list_str.strip().split():
        e, m = token.split(",")
        pairs.append((int(e), int(m)))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Analyze fake quantization distortion for random embedding vectors.")
    parser.add_argument("-n", "--num", type=int, default=10_000, help="Number of 1xD vectors (rows).")
    parser.add_argument("-d", "--embedding-sizes", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048], help="Embedding sizes to test.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--dist", choices=["normal", "uniform"], default="normal", help="Random distribution.")
    parser.add_argument("--std", type=float, default=1.0, help="Standard deviation of random vectors.")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type. NumPy: float16,float32,float64. Torch: float16,float32,float64,bfloat16.")
    parser.add_argument("--framework", choices=["torch", "numpy"], default="torch", help="Backend framework.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu/cuda). Ignored for NumPy.")
    parser.add_argument("--outdir", type=str, default="quantization_results", help="Output directory.")
    parser.add_argument("--no-csv", action="store_true", help="Do not save CSVs.")
    parser.add_argument("--chain", action="store_true",
                        help="Chain quantization across levels (measure vs original).")

    # Quantizer selection
    parser.add_argument("--quantizer", choices=["int", "em"], default="int", help="Quantization type.")

    # INT params
    parser.add_argument("--bits-from", type=int, default=8, help="INT: highest bit-width (inclusive).")
    parser.add_argument("--bits-to", type=int, default=3, help="INT: lowest bit-width (inclusive).")

    # EM params
    parser.add_argument("-e", "--em-exp", type=int, help="EM: exponent bits for a single format.")
    parser.add_argument("-m", "--em-man", type=int, help="EM: mantissa bits for a single format.")
    parser.add_argument("--em-list", type=str, help='EM: space-separated list like "4,3 5,2 5,3".')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Prepare x-axis levels in the intended plotting order
    if args.quantizer == "int":
        if args.bits_from < args.bits_to:
            raise SystemExit("--bits-from must be >= --bits-to")
        bits_list = list(range(args.bits_from, args.bits_to - 1, -1))  # e.g., [8,7,6,5,4,3]
        levels = [f"int{b}" for b in bits_list]
    else:
        if args.em_list:
            em_list = parse_em_list(args.em_list)
        else:
            if args.em_exp is None or args.em_man is None:
                raise SystemExit("For EM mode, specify either --em-list or both -e/--em-exp and -m/--em-man.")
            em_list = [(args.em_exp, args.em_man)]
        levels = [f"e{e}m{m}" for (e, m) in em_list]

    dfs_by_dim: Dict[int, pd.DataFrame] = {}

    if args.framework == "numpy":
        np_dtype = parse_numpy_dtype(args.dtype)
        for d in args.embedding_sizes:
            if args.quantizer == "int":
                print(f"[NumPy][INT] D={d} N={args.num} std={args.std} dtype={args.dtype} bits={levels} chain={args.chain}")
                df = np_run_trial_int(args.num, d, bits_list, args.seed, args.dist, args.chain, args.std, np_dtype)
            else:
                print(f"[NumPy][EM]  D={d} N={args.num} std={args.std} dtype={args.dtype} formats={levels} chain={args.chain}")
                df = np_run_trial_em(args.num, d, em_list, args.seed, args.dist, args.chain, args.std, np_dtype)
            dfs_by_dim[d] = df
            if not args.no_csv:
                df.to_csv(os.path.join(args.outdir, f"{args.quantizer}_stats_D{d}.csv"), index=False)
            plot_per_dim(df, d, args.outdir, args.quantizer)

    else:
        if torch is None:
            raise SystemExit("PyTorch not installed. Install torch or use --framework numpy.")
        th_dtype = parse_torch_dtype(args.dtype)
        device = args.device
        for d in args.embedding_sizes:
            if args.quantizer == "int":
                print(f"[Torch][INT] D={d} N={args.num} std={args.std} dtype={args.dtype} device={device} bits={levels} chain={args.chain}")
                df = th_run_trial_int(args.num, d, bits_list, args.seed, args.dist, args.chain, args.std, th_dtype, device)
            else:
                print(f"[Torch][EM]  D={d} N={args.num} std={args.std} dtype={args.dtype} device={device} formats={levels} chain={args.chain}")
                df = th_run_trial_em(args.num, d, em_list, args.seed, args.dist, args.chain, args.std, th_dtype, device)
            dfs_by_dim[d] = df
            if not args.no_csv:
                df.to_csv(os.path.join(args.outdir, f"{args.quantizer}_stats_D{d}.csv"), index=False)
            plot_per_dim(df, d, args.outdir, args.quantizer)

    # Combined plots (aligned to `levels`)
    plot_combined(dfs_by_dim, labels=levels, outdir=args.outdir, quantizer=args.quantizer)
    print(f"[Saved] {args.quantizer}_angle_deg_ALL.png, {args.quantizer}_cosine_ALL.png, {args.quantizer}_mag_delta_ALL.png in {args.outdir}")
    print("[Done]")


if __name__ == "__main__":
    main()

