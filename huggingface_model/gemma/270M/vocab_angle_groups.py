#!/usr/bin/env python3
"""Analyze Gemma vocabulary embedding angle graph and connected groups.

Builds an undirected graph over vocab tokens where an edge exists when the
pairwise angle is <= threshold_degrees. Produces CSV summaries, plots, and a
focused report for digit tokens 0-9.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma vocab angle component analysis")
    parser.add_argument("--model", default="google/gemma-3-270m", help="HF model name")
    parser.add_argument(
        "--embedding-source",
        choices=["input", "lm_head"],
        default="input",
        help="Use model input embeddings or lm_head weights",
    )
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=70.0,
        help="Create graph edge when pairwise angle <= this threshold",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=-1,
        help="Limit vocab size for analysis (-1 for full vocab)",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Block size for similarity scan")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--output-dir", default="./gemma_vocab_angle_groups")
    parser.add_argument(
        "--write-token-assignments",
        action="store_true",
        help="Write full token->component CSV (can be large)",
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _decode_tokens(tokenizer: AutoTokenizer, vocab_size: int) -> list[str]:
    tokens = [""] * vocab_size
    for idx in range(vocab_size):
        tokens[idx] = tokenizer.convert_ids_to_tokens(idx)
    return tokens


def _digit_token_map(tokens: list[str]) -> dict[str, list[int]]:
    digit_map: dict[str, list[int]] = {str(d): [] for d in range(10)}
    for idx, tok in enumerate(tokens):
        cleaned = tok.replace("▁", "").strip()
        if len(cleaned) == 1 and cleaned.isdigit():
            digit_map[cleaned].append(idx)
    return digit_map


def _compute_edges(
    emb: torch.Tensor,
    cosine_threshold: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = emb.size(0)
    src_all: list[np.ndarray] = []
    dst_all: list[np.ndarray] = []
    degrees = np.zeros(n, dtype=np.int64)

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        a = emb[i0:i1]
        for j0 in range(i0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            b = emb[j0:j1]
            sims = torch.matmul(a, b.T)
            mask = sims >= cosine_threshold
            if i0 == j0:
                tri = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
                mask = mask & tri

            pairs = mask.nonzero(as_tuple=False)
            if pairs.numel() == 0:
                continue

            src = (pairs[:, 0] + i0).cpu().numpy().astype(np.int32)
            dst = (pairs[:, 1] + j0).cpu().numpy().astype(np.int32)
            src_all.append(src)
            dst_all.append(dst)
            degrees += np.bincount(src, minlength=n)
            degrees += np.bincount(dst, minlength=n)

    if src_all:
        src_cat = np.concatenate(src_all)
        dst_cat = np.concatenate(dst_all)
    else:
        src_cat = np.empty((0,), dtype=np.int32)
        dst_cat = np.empty((0,), dtype=np.int32)
    return src_cat, dst_cat, degrees


def _write_component_summary(
    output_dir: Path,
    component_ids: np.ndarray,
    component_sizes: np.ndarray,
) -> None:
    counts = np.bincount(component_ids, minlength=component_sizes.shape[0])
    ordered = np.argsort(counts)[::-1]
    with (output_dir / "component_summary.csv").open("w", encoding="utf-8") as f:
        f.write("component_id,size\n")
        for cid in ordered:
            f.write(f"{cid},{int(counts[cid])}\n")


def _write_digit_reports(
    output_dir: Path,
    digit_map: dict[str, list[int]],
    component_ids: np.ndarray,
    component_sizes: np.ndarray,
    degrees: np.ndarray,
    tokens: list[str],
) -> None:
    rows: list[tuple[str, int, str, int, int, str]] = []
    for d in range(10):
        digit = str(d)
        ids = digit_map[digit]
        if not ids:
            rows.append((digit, 0, "", -1, 0, ""))
            continue
        for token_id in ids:
            comp = int(component_ids[token_id])
            rows.append(
                (
                    digit,
                    len(ids),
                    tokens[token_id],
                    comp,
                    int(component_sizes[comp]),
                    str(int(degrees[token_id])),
                )
            )

    with (output_dir / "digit_report.csv").open("w", encoding="utf-8") as f:
        f.write("digit,num_digit_tokens,token,component_id,component_size,degree\n")
        for row in rows:
            f.write(",".join(map(str, row)).replace("\n", " ") + "\n")

    with (output_dir / "digit_report.md").open("w", encoding="utf-8") as f:
        f.write("# Digit Token Report (0-9)\n\n")
        f.write("| Digit | Num Digit Tokens | Token | Component ID | Component Size | Degree |\n")
        f.write("|---|---:|---|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row[0]} | {row[1]} | `{row[2]}` | {row[3]} | {row[4]} | {row[5]} |\n"
            )


def _make_plots(output_dir: Path, component_sizes: np.ndarray, degrees: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    sorted_sizes = np.sort(component_sizes)[::-1]

    plt.figure(figsize=(10, 5))
    top_n = min(30, len(sorted_sizes))
    plt.bar(np.arange(top_n), sorted_sizes[:top_n])
    plt.title("Top Connected Component Sizes")
    plt.xlabel("Component Rank")
    plt.ylabel("Size")
    plt.tight_layout()
    plt.savefig(output_dir / "component_sizes_top30.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(component_sizes, bins=50, log=True)
    plt.title("Distribution of Connected Component Sizes")
    plt.xlabel("Component Size")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "component_size_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(degrees, bins=60, log=True)
    plt.title("Degree Distribution")
    plt.xlabel("Node Degree")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "degree_hist.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = _torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")

    if args.embedding_source == "input":
        emb = model.get_input_embeddings().weight.detach()
    else:
        emb = model.lm_head.weight.detach()

    if args.max_vocab > 0:
        emb = emb[: args.max_vocab]

    emb = emb.to(device=args.device, dtype=dtype)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    cosine_threshold = math.cos(math.radians(args.angle_threshold_deg))
    src, dst, degrees = _compute_edges(emb, cosine_threshold=cosine_threshold, chunk_size=args.chunk_size)

    n = emb.size(0)
    if src.size == 0:
        component_ids = np.arange(n, dtype=np.int32)
        component_sizes = np.ones(n, dtype=np.int32)
    else:
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components

            data = np.ones(src.shape[0] * 2, dtype=np.uint8)
            row = np.concatenate([src, dst])
            col = np.concatenate([dst, src])
            graph = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
            num_components, labels = connected_components(graph, directed=False, return_labels=True)
            component_ids = labels.astype(np.int32)
            component_sizes = np.bincount(component_ids, minlength=num_components).astype(np.int32)
        except ImportError as err:
            raise RuntimeError(
                "scipy is required to compute connected components when edges are present. "
                "Install scipy or rerun with a stricter angle threshold."
            ) from err

    tokens = _decode_tokens(tokenizer, n)
    digit_map = _digit_token_map(tokens)

    with (output_dir / "analysis_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"embedding_source={args.embedding_source}\n")
        f.write(f"vocab_size_analyzed={n}\n")
        f.write(f"angle_threshold_deg={args.angle_threshold_deg}\n")
        f.write(f"cosine_threshold={cosine_threshold:.8f}\n")
        f.write(f"num_edges={int(src.size)}\n")
        f.write(f"num_components={int(component_sizes.shape[0])}\n")
        f.write(f"largest_component={int(component_sizes.max())}\n")
        f.write(f"isolated_tokens={(degrees == 0).sum()}\n")

    _write_component_summary(output_dir, component_ids, component_sizes)
    _write_digit_reports(output_dir, digit_map, component_ids, component_sizes, degrees, tokens)

    if args.write_token_assignments:
        with (output_dir / "token_component_assignments.csv").open("w", encoding="utf-8") as f:
            f.write("token_id,token,component_id,component_size,degree\n")
            for token_id in range(n):
                token = tokens[token_id].replace("\n", "\\n")
                cid = int(component_ids[token_id])
                f.write(f"{token_id},{token},{cid},{int(component_sizes[cid])},{int(degrees[token_id])}\n")

    _make_plots(output_dir, component_sizes, degrees)
    print(f"Done. Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
