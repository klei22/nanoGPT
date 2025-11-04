"""Vector lookup table quantization for nanoGPT checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from typing import Dict, Tuple

import numpy as np
import torch

from .lut_generators import (
    GAUSSIAN_BASELINE_METHOD,
    HALTON_METHOD,
    KRONECKER_METHOD,
    RANDOM_SPHERE_METHOD,
    RSEQ_METHOD,
    build_unit_lut,
)

DEFAULT_METHOD = KRONECKER_METHOD
DEFAULT_DIM = 384
DEFAULT_CHUNK = 16384
DEFAULT_ROW_CHUNK = 512
DEFAULT_LUT_OUT = "vector_lut.npy"
DEFAULT_INDICES_SUBDIR = "vector_lut_indices"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize a checkpoint by snapping rows of tensors to vectors from a "
            "pre-generated lookup table on the hypersphere."
        )
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and metadata from a training run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write the quantized checkpoint (defaults to <ckpt_dir>_<method>_lut)",
    )
    parser.add_argument(
        "--lut-method",
        type=str,
        default=DEFAULT_METHOD,
        choices=(
            KRONECKER_METHOD,
            HALTON_METHOD,
            RSEQ_METHOD,
            RANDOM_SPHERE_METHOD,
            GAUSSIAN_BASELINE_METHOD,
        ),
        help="Lookup table construction to use",
    )
    parser.add_argument(
        "--lut-size",
        type=int,
        required=True,
        help="Number of vectors in the lookup table",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help="Target feature dimension (matches n_embd for transformer weights)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic LUT generation where applicable",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help="Number of LUT vectors to process per block when scoring similarities",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=DEFAULT_ROW_CHUNK,
        help="Number of tensor rows to evaluate at once when scanning the LUT",
    )
    parser.add_argument(
        "--no-rescale",
        action="store_false",
        dest="rescale",
        help="Do not restore original row norms after snapping to LUT vectors",
    )
    parser.add_argument(
        "--gaussian-std",
        type=float,
        default=0.02,
        help="Standard deviation for the gaussian baseline LUT (mean fixed at 0.0)",
    )
    parser.add_argument(
        "--lut-out",
        type=str,
        default=DEFAULT_LUT_OUT,
        help="Filename (or path) for storing the LUT as .npy; set to '' to skip saving",
    )
    parser.add_argument(
        "--indices-out",
        type=str,
        default=DEFAULT_INDICES_SUBDIR,
        help=(
            "Directory (relative to out_dir unless absolute) for saving per-tensor "
            "LUT indices; set to '' to skip saving"
        ),
    )
    parser.add_argument(
        "--no-copy-extras",
        action="store_true",
        help="Do not copy auxiliary files from ckpt_dir into out_dir",
    )
    parser.set_defaults(rescale=True)
    return parser.parse_args()


def _resolve_out_dir(args: argparse.Namespace) -> str:
    if args.out_dir:
        return args.out_dir
    suffix = f"{args.lut_method}_lut_{args.lut_size}"
    return os.path.abspath(os.path.join(args.ckpt_dir, f"{suffix}"))


def _resolve_path(base: str, target: str | None) -> str | None:
    if not target:
        return None
    if os.path.isabs(target):
        return target
    return os.path.join(base, target)


def _copy_aux_files(src_dir: str, dst_dir: str) -> None:
    dst_dir_abs = os.path.abspath(dst_dir)
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [
            d
            for d in dirs
            if os.path.abspath(os.path.join(root, d)) != dst_dir_abs
        ]
        rel = os.path.relpath(root, src_dir)
        dst_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(dst_root, exist_ok=True)
        for fname in files:
            if fname == "ckpt.pt":
                continue
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_root, fname)
            if os.path.isdir(src_path):
                continue
            shutil.copy2(src_path, dst_path)


def _quantize_tensor(
    tensor: torch.Tensor,
    lut: torch.Tensor,
    *,
    dim: int,
    chunk_size: int,
    row_chunk_size: int,
    rescale: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = tensor.shape
    flat = tensor.detach().cpu().to(torch.float32)
    flat = flat.reshape(-1, dim)
    norms = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    unit_rows = flat / norms

    chunk = max(1, min(chunk_size, lut.size(0)))
    row_chunk = max(1, min(row_chunk_size, unit_rows.size(0)))
    best_scores = torch.full((unit_rows.size(0),), -math.inf, dtype=torch.float32)
    best_indices = torch.zeros((unit_rows.size(0),), dtype=torch.long)

    for row_start in range(0, unit_rows.size(0), row_chunk):
        row_stop = min(row_start + row_chunk, unit_rows.size(0))
        row_block = unit_rows[row_start:row_stop]
        block_scores = torch.full((row_block.size(0),), -math.inf, dtype=torch.float32)
        block_indices = torch.zeros((row_block.size(0),), dtype=torch.long)
        for start in range(0, lut.size(0), chunk):
            stop = min(start + chunk, lut.size(0))
            scores = row_block @ lut[start:stop].T
            values, local_idx = scores.max(dim=1)
            improved = values > block_scores
            if torch.any(improved):
                block_scores[improved] = values[improved]
                block_indices[improved] = local_idx[improved] + start
        best_scores[row_start:row_stop] = block_scores
        best_indices[row_start:row_stop] = block_indices

    snapped = lut[best_indices]
    if rescale:
        snapped = snapped * norms
    snapped = snapped.reshape(original_shape).to(tensor.dtype)
    return snapped, best_indices, best_scores


def _quantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    lut: torch.Tensor,
    *,
    dim: int,
    chunk_size: int,
    row_chunk_size: int,
    rescale: bool,
    indices_dir: str | None,
) -> Dict[str, Dict[str, float | int | str | None]]:
    stats: Dict[str, Dict[str, float | int | str | None]] = {}
    if indices_dir:
        os.makedirs(indices_dir, exist_ok=True)

    for name, tensor in list(state_dict.items()):
        if not torch.is_tensor(tensor):
            continue
        if tensor.numel() == 0:
            continue
        if tensor.shape[-1] != dim:
            continue
        snapped, indices, scores = _quantize_tensor(
            tensor,
            lut,
            dim=dim,
            chunk_size=chunk_size,
            row_chunk_size=row_chunk_size,
            rescale=rescale,
        )
        state_dict[name] = snapped
        record: Dict[str, float | int | str | None] = {
            "entries": int(indices.numel()),
            "mean_similarity": float(scores.mean().item()),
            "min_similarity": float(scores.min().item()),
            "max_similarity": float(scores.max().item()),
            "indices_path": None,
        }
        if indices_dir:
            safe_name = name.replace("/", "_").replace(".", "_")
            out_path = os.path.join(indices_dir, f"{safe_name}_indices.npy")
            np.save(out_path, indices.numpy())
            record["indices_path"] = out_path
        stats[name] = record
    return stats


def main() -> None:
    args = _parse_args()
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    out_dir = os.path.abspath(_resolve_out_dir(args))
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not args.no_copy_extras:
        _copy_aux_files(ckpt_dir, out_dir)

    lut = build_unit_lut(
        method=args.lut_method,
        num_vectors=args.lut_size,
        dim=args.dim,
        seed=args.seed,
        gaussian_std=args.gaussian_std,
    )
    lut_tensor = torch.from_numpy(lut)

    lut_out_path = _resolve_path(out_dir, args.lut_out)
    if lut_out_path:
        os.makedirs(os.path.dirname(lut_out_path), exist_ok=True)
        np.save(lut_out_path, lut)

    indices_dir = _resolve_path(out_dir, args.indices_out)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    target_state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if target_state is None:
        if isinstance(ckpt, dict):
            target_state = ckpt
        else:
            raise ValueError("Unsupported checkpoint format")

    stats = _quantize_state_dict(
        target_state,
        lut_tensor,
        dim=args.dim,
        chunk_size=args.chunk_size,
        row_chunk_size=args.row_chunk_size,
        rescale=args.rescale,
        indices_dir=indices_dir,
    )

    metadata = {
        "method": args.lut_method,
        "lut_size": args.lut_size,
        "dim": args.dim,
        "seed": args.seed,
        "rescale": args.rescale,
        "chunk_size": args.chunk_size,
        "gaussian_std": args.gaussian_std,
        "lut_path": lut_out_path,
        "indices_dir": indices_dir,
        "tensor_stats": stats,
    }

    ckpt_out_path = os.path.join(out_dir, "ckpt.pt")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            ckpt["model"] = target_state
        ckpt.setdefault("quantization", {})["vector_lut"] = metadata
    torch.save(ckpt, ckpt_out_path)

    meta_path = os.path.join(out_dir, "vector_lut_quantization.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Wrote vector-LUT quantized checkpoint to {ckpt_out_path}")
    print(f"Metadata saved to {meta_path}")
    if lut_out_path:
        print(f"Lookup table saved to {lut_out_path}")
    if indices_dir:
        print(f"Per-tensor LUT indices saved under {indices_dir}")


if __name__ == "__main__":
    main()
