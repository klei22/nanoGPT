"""Vector lookup quantization for checkpoint weights.

This script implements a vector-quantization style post-training
quantization (PTQ) scheme that replaces each eligible weight vector with the
closest entry from a hyperspherical codebook.  Instead of snapping to a
uniform fake quantization grid, we project the direction of each vector onto
the codebook entry with the highest cosine similarity and then restore the
original norm.  The codebook itself can be generated using several of the
high-dimensional constructions that already exist in the analysis utilities
(`analysis/hypersphere_grid/hypersphere_grid.py` and
`analysis/hypersphere_lattice/hypersphere_lattices.py`).

The resulting checkpoint preserves the original tensor structure and dtype,
but eligible tensors whose trailing dimension matches the embedding size
(384 by default) are replaced with their vector-lookup approximations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helper utilities to interoperate with the existing PTQ code.
# ---------------------------------------------------------------------------


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{module_name}' from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRID_PATH = _REPO_ROOT / "analysis" / "hypersphere_grid" / "hypersphere_grid.py"
_LATTICE_PATH = _REPO_ROOT / "analysis" / "hypersphere_lattice" / "hypersphere_lattices.py"

_GRID_MODULE = _load_module_from_path("analysis.hypersphere_grid", _GRID_PATH)
_LATTICE_MODULE = _load_module_from_path(
    "analysis.hypersphere_lattices", _LATTICE_PATH
)


def iter_state_items(state_dict) -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(state_dict, torch.nn.Module):
        iterable = state_dict.state_dict().items()
    elif isinstance(state_dict, dict):
        iterable = state_dict.items()
    else:
        iterable = getattr(state_dict, "state_dict", lambda: {})().items()

    for key, value in iterable:
        if torch.is_tensor(value):
            yield key, value


# ---------------------------------------------------------------------------
# Codebook generation helpers.
# ---------------------------------------------------------------------------


def _normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def _codebook_from_grid(method: str, num_vectors: int, dim: int, seed: int) -> np.ndarray:
    if method == "kronecker":
        return _GRID_MODULE.kronecker_sphere(num_vectors, dim, seed=seed)
    if method == "halton":
        return _GRID_MODULE.halton_sphere(num_vectors, dim)
    if method == "random":
        return _GRID_MODULE.random_sphere(num_vectors, dim, seed=seed)
    raise ValueError(f"Unknown hypersphere grid method: {method}")


def _codebook_from_lattice(method: str, num_vectors: int, dim: int, seed: int) -> np.ndarray:
    return _LATTICE_MODULE.generate_points(num_vectors, dim, method, seed=seed)


def build_codebook(
    method: str,
    num_vectors: int,
    dim: int,
    seed: int,
) -> np.ndarray:
    if num_vectors <= 0:
        raise ValueError("Number of codebook vectors must be positive")

    if method.startswith("grid_"):
        return _codebook_from_grid(method.split("_", 1)[1], num_vectors, dim, seed)
    if method.startswith("lattice_"):
        return _codebook_from_lattice(method.split("_", 1)[1], num_vectors, dim, seed)
    if method == "gaussian_baseline":
        rng = np.random.default_rng(seed)
        samples = rng.normal(loc=0.0, scale=0.02, size=(num_vectors, dim))
        return _normalize_rows(samples)
    raise ValueError(f"Unknown codebook method: {method}")


# ---------------------------------------------------------------------------
# Lookup quantization core.
# ---------------------------------------------------------------------------


@dataclass
class QuantizationStats:
    replaced_tensors: List[str]
    skipped_tensors: List[str]
    total_rows: int
    matched_rows: int


def _lookup_replace(
    matrix: torch.Tensor,
    codebook: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, int]:
    if matrix.numel() == 0:
        return matrix, 0

    orig_shape = matrix.shape
    flat = matrix.view(-1, matrix.shape[-1])
    norms = flat.norm(dim=1, keepdim=True)
    nonzero = norms.squeeze(1) > 0
    normalized = torch.zeros_like(flat)
    normalized[nonzero] = flat[nonzero] / norms[nonzero]

    num_rows = flat.shape[0]
    best_sim = torch.full((num_rows,), -math.inf, device=flat.device, dtype=flat.dtype)
    best_idx = torch.full((num_rows,), -1, device=flat.device, dtype=torch.long)

    start = 0
    while start < codebook.shape[0]:
        end = min(start + chunk_size, codebook.shape[0])
        chunk = codebook[start:end]
        sims = torch.matmul(normalized, chunk.t())
        values, indices = sims.max(dim=1)
        better = values > best_sim
        best_sim[better] = values[better]
        best_idx[better] = indices[better] + start
        start = end

    selected = codebook[best_idx]
    reconstructed = selected * norms
    reconstructed = reconstructed.view(orig_shape)
    matched = int(nonzero.sum().item())
    return reconstructed.to(matrix.dtype), matched


def vector_lookup_quantize(
    tensor: torch.Tensor,
    codebook: torch.Tensor,
    dim: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, int]:
    if not torch.is_floating_point(tensor):
        return tensor, 0
    if tensor.shape[-1] != dim:
        return tensor, 0

    device = tensor.device
    working = tensor.to(torch.float32, copy=True)
    codebook = codebook.to(device=device, dtype=working.dtype)
    quantized, matched = _lookup_replace(working, codebook, chunk_size)
    return quantized.to(tensor.dtype), matched


def quantize_state_dict(
    state_dict,
    codebook: torch.Tensor,
    dim: int,
    chunk_size: int,
) -> QuantizationStats:
    replaced: List[str] = []
    skipped: List[str] = []
    total_rows = 0
    matched_rows = 0

    for name, tensor in iter_state_items(state_dict):
        if not torch.is_floating_point(tensor):
            continue
        if tensor.shape[-1] != dim:
            skipped.append(name)
            continue

        flat_rows = tensor.numel() // dim
        total_rows += flat_rows

        quantized, matched = vector_lookup_quantize(tensor, codebook, dim, chunk_size)
        state_dict[name] = quantized
        replaced.append(name)
        matched_rows += matched

    return QuantizationStats(
        replaced_tensors=replaced,
        skipped_tensors=skipped,
        total_rows=total_rows,
        matched_rows=matched_rows,
    )


# ---------------------------------------------------------------------------
# Command-line interface.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize checkpoint tensors by replacing weight rows with the closest "
            "vector from a hyperspherical codebook."
        )
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and meta.pkl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for the quantized checkpoint (defaults to <ckpt_dir>_lookup_ptq)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid_kronecker",
        choices=[
            "grid_kronecker",
            "grid_halton",
            "grid_random",
            "lattice_rseq",
            "lattice_halton",
            "lattice_random",
            "gaussian_baseline",
        ],
        help="Codebook construction method",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=10000,
        help="Number of codebook vectors to generate",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Embedding dimension for eligible tensor rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stochastic codebook generation",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=65536,
        help="Number of codebook entries to process per chunk during lookup",
    )
    parser.add_argument(
        "--save_codebook",
        action="store_true",
        help="Save the generated codebook (NumPy .npy) alongside the checkpoint",
    )
    parser.add_argument(
        "--metadata_filename",
        type=str,
        default="lookup_ptq_metadata.json",
        help="Filename for JSON metadata describing the quantization run",
    )
    return parser.parse_args()


def _resolve_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_obj = checkpoint["model"]
    else:
        state_obj = checkpoint

    if isinstance(state_obj, dict):
        return checkpoint, state_obj

    to_state = getattr(state_obj, "state_dict", None)
    if callable(to_state):
        state_dict = to_state()
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint["model"] = state_dict
        else:
            checkpoint = state_dict
        return checkpoint, state_dict

    raise TypeError("Unsupported checkpoint format: expected mapping-like state")


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk_size must be positive")
    ckpt_path = Path(args.ckpt_dir) / "ckpt.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    checkpoint, state_dict = _resolve_state_dict(checkpoint)

    codebook_np = build_codebook(args.method, args.num_vectors, args.dim, args.seed)
    codebook = torch.from_numpy(_normalize_rows(codebook_np.astype(np.float32)))

    stats = quantize_state_dict(state_dict, codebook, args.dim, args.chunk_size)

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{args.ckpt_dir}_lookup_ptq")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "ckpt.pt"
    torch.save(checkpoint, str(out_path))

    metadata: Dict[str, object] = {
        "method": args.method,
        "num_vectors": args.num_vectors,
        "dim": args.dim,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "replaced_tensors": stats.replaced_tensors,
        "skipped_tensors": stats.skipped_tensors,
        "total_rows": stats.total_rows,
        "matched_rows": stats.matched_rows,
    }

    meta_path = out_dir / args.metadata_filename
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    if args.save_codebook:
        np.save(out_dir / "codebook.npy", codebook.numpy())

    print(
        f"Vector lookup PTQ complete. Saved checkpoint to {out_path}.\n"
        f"Replaced {len(stats.replaced_tensors)} tensors (rows matched: {stats.matched_rows}/"
        f"{stats.total_rows})."
    )


if __name__ == "__main__":
    main()

