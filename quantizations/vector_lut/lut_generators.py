"""Lookup table generators for vector-based quantization.

The implementations mirror the high-dimensional constructions provided in the
``analysis/hypersphere_grid`` and ``analysis/hypersphere_lattice`` utilities so
that demo scripts can easily compare coverage properties across generators.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

KRONECKER_METHOD: Literal["kronecker"] = "kronecker"
HALTON_METHOD: Literal["halton"] = "halton"
RSEQ_METHOD: Literal["rseq"] = "rseq"
RANDOM_SPHERE_METHOD: Literal["random_sphere"] = "random_sphere"
GAUSSIAN_BASELINE_METHOD: Literal["gaussian_baseline"] = "gaussian_baseline"

_SUPPORTED_METHODS = {
    KRONECKER_METHOD,
    HALTON_METHOD,
    RSEQ_METHOD,
    RANDOM_SPHERE_METHOD,
    GAUSSIAN_BASELINE_METHOD,
}


def _unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _inv_norm_cdf(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF using Acklam's rational approximation."""

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ],
        dtype=np.float64,
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ],
        dtype=np.float64,
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ],
        dtype=np.float64,
    )

    p = np.asarray(p, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1.0 - eps)

    lower = p < 0.02425
    upper = p > 1.0 - 0.02425
    middle = ~(lower | upper)

    result = np.empty_like(p)
    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        result[lower] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        result[middle] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        result[upper] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    return result


def _cube_to_gaussian(u01: np.ndarray) -> np.ndarray:
    return _inv_norm_cdf(u01)


def _cube_to_sphere(u01: np.ndarray) -> np.ndarray:
    return _unit_rows(_cube_to_gaussian(u01))


def _prime_sieve(limit: int) -> np.ndarray:
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i : limit + 1 : i] = False
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


def _halton_sequence(dim: int, n: int, start: int = 1) -> np.ndarray:
    bases = _first_primes(dim)
    seq = np.empty((n, dim), dtype=np.float64)
    for j, base in enumerate(bases):
        f = 1.0
        r = 0.0
        for i in range(n):
            index = start + i
            f = 1.0
            r = 0.0
            k = index
            inv_base = 1.0 / base
            place = inv_base
            while k > 0:
                digit = k % base
                r += digit * place
                k //= base
                place *= inv_base
            seq[i, j] = r
    return seq


def _kronecker_sequence(dim: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shift = rng.uniform(0.0, 1.0)
    primes = _first_primes(dim)
    alphas = np.sqrt(primes.astype(np.float64) + shift)
    alphas = np.modf(alphas)[0]
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    offsets = np.array([math.modf(phi ** (i + 1))[0] for i in range(dim)], dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)[:, None]
    seq = (k * alphas[None, :] + offsets[None, :]) % 1.0
    return seq


def _rseq_sequence(dim: int, n: int) -> np.ndarray:
    primes = _first_primes(dim)
    alphas = np.sqrt(primes.astype(np.float64))
    alphas -= np.floor(alphas)
    k = np.arange(n, dtype=np.float64)[:, None]
    offsets = np.full(dim, 0.5, dtype=np.float64)
    seq = (offsets[None, :] + k * alphas[None, :]) % 1.0
    return seq


def build_unit_lut(
    method: str,
    num_vectors: int,
    dim: int,
    *,
    seed: int = 0,
    gaussian_std: float = 0.02,
) -> np.ndarray:
    """Construct a unit-normalized lookup table for the requested method."""

    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported LUT method: {method}")
    if num_vectors <= 0:
        raise ValueError("num_vectors must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    if method == RANDOM_SPHERE_METHOD:
        rng = np.random.default_rng(seed)
        samples = rng.standard_normal(size=(num_vectors, dim))
        vectors = _unit_rows(samples)
    elif method == GAUSSIAN_BASELINE_METHOD:
        rng = np.random.default_rng(seed)
        samples = rng.normal(loc=0.0, scale=gaussian_std, size=(num_vectors, dim))
        vectors = _unit_rows(samples)
    elif method == HALTON_METHOD:
        seq = _halton_sequence(dim, num_vectors, start=1)
        vectors = _cube_to_sphere(seq)
    elif method == KRONECKER_METHOD:
        seq = _kronecker_sequence(dim, num_vectors, seed=seed)
        vectors = _cube_to_sphere(seq)
    elif method == RSEQ_METHOD:
        seq = _rseq_sequence(dim, num_vectors)
        vectors = _cube_to_sphere(seq)
    else:  # pragma: no cover - defensive, should not be reachable
        raise RuntimeError(f"Unreachable method dispatch for {method}")

    return vectors.astype(np.float32, copy=False)


__all__ = [
    "build_unit_lut",
    "KRONECKER_METHOD",
    "HALTON_METHOD",
    "RSEQ_METHOD",
    "RANDOM_SPHERE_METHOD",
    "GAUSSIAN_BASELINE_METHOD",
]
