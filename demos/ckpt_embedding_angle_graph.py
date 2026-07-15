#!/usr/bin/env python3
"""Export a checkpoint token-embedding angle graph as CSVs for the fast viewer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_labels(meta_path: Path | None, vocab_size: int) -> list[str]:
    labels = [str(i) for i in range(vocab_size)]
    if meta_path is None or not meta_path.exists():
        return labels
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    itos = meta.get("itos") if isinstance(meta, dict) else None
    if isinstance(itos, dict):
        for idx in range(vocab_size):
            value = itos.get(idx, itos.get(str(idx)))
            if value is not None:
                labels[idx] = str(value)
    elif isinstance(itos, (list, tuple)):
        for idx, value in enumerate(itos[:vocab_size]):
            labels[idx] = str(value)
    return labels


def load_embedding(ckpt_path: Path, source: str) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    key = "transformer.wte.weight" if source == "wte" else "lm_head.weight"
    tensor = state.get(key)
    if tensor is None:
        tensor = state.get(f"_orig_mod.{key}")
    if tensor is None:
        raise KeyError(f"Could not find {key} in {ckpt_path}")
    return tensor.detach().float()


def build_edges(emb: torch.Tensor, top_k: int, max_angle_deg: float | None) -> dict[tuple[int, int], float]:
    emb = F.normalize(emb, dim=-1, eps=1e-12)
    sim = (emb @ emb.T).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(sim))
    angles.fill_diagonal_(float("inf"))
    edges: dict[tuple[int, int], float] = {}
    if top_k > 0:
        k = min(top_k, max(0, emb.shape[0] - 1))
        values, indices = torch.topk(angles, k=k, dim=1, largest=False)
        for source in range(emb.shape[0]):
            for angle, target in zip(values[source].tolist(), indices[source].tolist()):
                if max_angle_deg is not None and angle > max_angle_deg:
                    continue
                pair = tuple(sorted((source, int(target))))
                edges[pair] = min(float(angle), edges.get(pair, float("inf")))
    elif max_angle_deg is not None:
        rows, cols = torch.where(torch.triu(angles <= max_angle_deg, diagonal=1))
        for source, target in zip(rows.tolist(), cols.tolist()):
            edges[(int(source), int(target))] = float(angles[source, target].item())
    else:
        raise ValueError("Set --top-k or --max-angle-deg so the graph is not complete by default.")
    return edges


def write_outputs(
    output_dir: Path,
    labels: list[str],
    emb: torch.Tensor,
    edges: dict[tuple[int, int], float],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = len(labels)
    degree = [0] * vocab_size
    for source, target in edges:
        degree[source] += 1
        degree[target] += 1

    token_csv = output_dir / "token_list.csv"
    with token_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Token ID", "Raw token", "Display", "Connected nodes", "Vector length"])
        norms = emb.norm(dim=-1).tolist()
        for idx, label in enumerate(labels):
            display = label.replace("\n", "\\n").replace("\t", "\\t")
            writer.writerow([idx, label, display, degree[idx], f"{norms[idx]:.8f}"])

    adjacency_csv = output_dir / "adjacency.csv"
    edge_lookup = {pair: angle for pair, angle in edges.items()}
    with adjacency_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", *range(vocab_size)])
        for row in range(vocab_size):
            values = [row]
            for col in range(vocab_size):
                if row == col:
                    values.append("0")
                else:
                    angle = edge_lookup.get(tuple(sorted((row, col))))
                    values.append("" if angle is None else f"{angle:.6f}")
            writer.writerow(values)

    dictionary = {
        str(idx): {
            "token_id": idx,
            "token_raw": labels[idx],
            "token_display": labels[idx].replace("\n", "\\n").replace("\t", "\\t"),
            "connected_count": degree[idx],
            "magnitude": float(emb[idx].norm().item()),
        }
        for idx in range(vocab_size)
    }
    (output_dir / "dictionary.json").write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps({**metadata, "nodes": vocab_size, "edges": len(edges)}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build token-embedding angle graph CSVs from a nanoGPT checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/shakespeare_char/meta.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=["wte", "lm_head"], default="wte")
    parser.add_argument("--top-k", type=int, default=4, help="Connect each token to its k nearest angular neighbors.")
    parser.add_argument("--max-angle-deg", type=float, default=None, help="Optional angle cutoff; combines with --top-k when both are set.")
    args = parser.parse_args()

    emb = load_embedding(args.ckpt, args.source)
    labels = load_labels(args.meta, emb.shape[0])
    edges = build_edges(emb, args.top_k, args.max_angle_deg)
    write_outputs(
        args.output_dir,
        labels,
        emb,
        edges,
        {
            "checkpoint": str(args.ckpt),
            "source": args.source,
            "top_k": args.top_k,
            "max_angle_deg": args.max_angle_deg,
        },
    )
    print(f"Wrote {args.output_dir} with {len(labels)} nodes and {len(edges)} edges")


if __name__ == "__main__":
    main()
