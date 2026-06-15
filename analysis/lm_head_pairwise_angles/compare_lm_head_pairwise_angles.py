#!/usr/bin/env python3
"""Compare pairwise lm_head vocabulary-vector angles between two checkpoints."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AngleComparison:
    ckpt_a: str
    ckpt_b: str
    lm_head_key_a: str
    lm_head_key_b: str
    tokens: List[str]
    angles_a: torch.Tensor
    angles_b: torch.Tensor
    diff: torch.Tensor
    mask: torch.Tensor
    pair_indices: torch.Tensor
    metrics: Dict[str, float]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return device


def load_checkpoint(path: str, device: str) -> Dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def clean_state_dict(ckpt: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}


def find_lm_head_weight(sd: Dict[str, torch.Tensor], key: Optional[str] = None) -> Tuple[str, torch.Tensor]:
    candidates = [key] if key else [
        "lm_head.weight",
        "transformer.lm_head_0.weight",
        "transformer.wte.weight",  # tied-weight checkpoints may carry only the embedding copy
    ]
    for name in candidates:
        if name and name in sd and sd[name].ndim == 2:
            return name, sd[name]
    suffix_matches = [name for name, value in sd.items() if name.endswith("lm_head.weight") and value.ndim == 2]
    suffix_matches += [name for name, value in sd.items() if "lm_head_" in name and name.endswith(".weight") and value.ndim == 2]
    if suffix_matches:
        name = sorted(suffix_matches)[0]
        return name, sd[name]
    raise KeyError("Could not locate a 2D lm_head weight in checkpoint state_dict")


def load_tokens(meta_path: Optional[str], vocab_size: int) -> List[str]:
    if not meta_path:
        return [str(i) for i in range(vocab_size)]
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    itos = meta.get("itos")
    if isinstance(itos, dict):
        return [str(itos.get(i, i)) for i in range(vocab_size)]
    if isinstance(itos, list):
        return [str(itos[i]) if i < len(itos) else str(i) for i in range(vocab_size)]
    return [str(i) for i in range(vocab_size)]


def pairwise_angles_deg(weight: torch.Tensor) -> torch.Tensor:
    vecs = F.normalize(weight.float(), p=2, dim=1, eps=1e-12)
    cosine = (vecs @ vecs.T).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def summarize(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    if values.numel() == 0:
        return {f"{prefix}_{k}": math.nan for k in ["mean", "median", "std", "min", "max"]}
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_median": float(values.median().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def compare_lm_head_pairwise_angles(
    ckpt_a: str,
    ckpt_b: str,
    *,
    meta: Optional[str] = None,
    device: str = "cpu",
    lm_head_key_a: Optional[str] = None,
    lm_head_key_b: Optional[str] = None,
    min_angle: float = 0.0,
    max_angle: float = 180.0,
) -> AngleComparison:
    device = resolve_device(device)
    sd_a = clean_state_dict(load_checkpoint(ckpt_a, device))
    sd_b = clean_state_dict(load_checkpoint(ckpt_b, device))
    key_a, weight_a = find_lm_head_weight(sd_a, lm_head_key_a)
    key_b, weight_b = find_lm_head_weight(sd_b, lm_head_key_b)
    if weight_a.shape != weight_b.shape:
        raise ValueError(f"lm_head shapes differ: {tuple(weight_a.shape)} vs {tuple(weight_b.shape)}")

    angles_a = pairwise_angles_deg(weight_a)
    angles_b = pairwise_angles_deg(weight_b)
    diff = angles_b - angles_a
    vocab_size = weight_a.shape[0]
    pair_indices = torch.triu_indices(vocab_size, vocab_size, offset=1, device=angles_a.device)
    a_pairs = angles_a[pair_indices[0], pair_indices[1]]
    b_pairs = angles_b[pair_indices[0], pair_indices[1]]
    diff_pairs = diff[pair_indices[0], pair_indices[1]]
    mask = (a_pairs >= min_angle) & (a_pairs <= max_angle)
    selected_a = a_pairs[mask]
    selected_b = b_pairs[mask]
    selected_diff = diff_pairs[mask]

    metrics: Dict[str, float] = {
        "vocab_size": float(vocab_size),
        "embedding_dim": float(weight_a.shape[1]),
        "total_pairs": float(a_pairs.numel()),
        "selected_pairs": float(selected_a.numel()),
        "angle_window_min": float(min_angle),
        "angle_window_max": float(max_angle),
    }
    metrics.update(summarize(selected_a, "ckpt_a_angle_deg"))
    metrics.update(summarize(selected_b, "ckpt_b_angle_deg"))
    metrics.update(summarize(selected_diff, "diff_deg"))
    if selected_diff.numel():
        metrics["mae_deg"] = float(selected_diff.abs().mean().item())
        metrics["rmse_deg"] = float(torch.sqrt((selected_diff ** 2).mean()).item())
        if selected_a.numel() > 1 and selected_a.std() > 0 and selected_b.std() > 0:
            metrics["pearson_r"] = float(torch.corrcoef(torch.stack([selected_a, selected_b]))[0, 1].item())
        else:
            metrics["pearson_r"] = math.nan
    return AngleComparison(ckpt_a, ckpt_b, key_a, key_b, load_tokens(meta, vocab_size), angles_a.cpu(), angles_b.cpu(), diff.cpu(), mask.cpu(), pair_indices.cpu(), metrics)


def selected_pair_rows(result: AngleComparison) -> Iterable[Dict[str, object]]:
    rows, cols = result.pair_indices
    selected = result.mask.nonzero(as_tuple=False).flatten()
    for order, idx in enumerate(selected.tolist()):
        i, j = int(rows[idx]), int(cols[idx])
        yield {
            "pair_order": order, "token_i": i, "token_j": j,
            "label_i": result.tokens[i], "label_j": result.tokens[j],
            "angle_a_deg": float(result.angles_a[i, j]),
            "angle_b_deg": float(result.angles_b[i, j]),
            "diff_deg": float(result.diff[i, j]),
        }


def write_csv(result: AngleComparison, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rows = list(selected_pair_rows(result))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["pair_order", "token_i", "token_j", "label_i", "label_j", "angle_a_deg", "angle_b_deg", "diff_deg"])
        writer.writeheader(); writer.writerows(rows)


def plot_html(result: AngleComparison) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = list(selected_pair_rows(result))
    x = [f"{r['token_i']}-{r['token_j']}" for r in rows]
    a = [r["angle_a_deg"] for r in rows]; b = [r["angle_b_deg"] for r in rows]; d = [r["diff_deg"] for r in rows]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Selected pair angles by canonical vocab-pair order", "Difference histogram", "Checkpoint A pairwise angles", "Angle difference (B - A)"), specs=[[{}, {}], [{"type": "heatmap"}, {"type": "heatmap"}]])
    fig.add_trace(go.Scatter(x=x, y=a, mode="markers", name="ckpt A", marker={"size": 5}), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=b, mode="markers", name="ckpt B", marker={"size": 5}), 1, 1)
    fig.add_trace(go.Histogram(x=d, name="B - A diff"), 1, 2)
    labels = [f"{i}:{t}" for i, t in enumerate(result.tokens)]
    fig.add_trace(go.Heatmap(z=result.angles_a.tolist(), x=labels, y=labels, colorscale="Viridis", colorbar={"title":"deg"}), 2, 1)
    fig.add_trace(go.Heatmap(z=result.diff.tolist(), x=labels, y=labels, colorscale="RdBu", zmid=0, colorbar={"title":"deg"}), 2, 2)
    fig.update_layout(height=1000, title="LM head pairwise angle comparison", bargap=0.05)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def write_html(result: AngleComparison, path: str) -> None:
    metrics = "".join(f"<tr><th>{html.escape(k)}</th><td>{v:.6g}</td></tr>" for k, v in result.metrics.items())
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>LM head pairwise angle comparison</title></head><body>
<h1>LM head pairwise angle comparison</h1>
<p><b>A:</b> {html.escape(result.ckpt_a)} ({html.escape(result.lm_head_key_a)})<br><b>B:</b> {html.escape(result.ckpt_b)} ({html.escape(result.lm_head_key_b)})</p>
<table border='1' cellpadding='4'>{metrics}</table>{plot_html(result)}</body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write(body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt_a"); p.add_argument("ckpt_b")
    p.add_argument("--meta", default=None); p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--lm-head-key-a", default=None); p.add_argument("--lm-head-key-b", default=None)
    p.add_argument("--min-angle", type=float, default=0.0); p.add_argument("--max-angle", type=float, default=180.0)
    p.add_argument("--csv", default=None); p.add_argument("--html", default=None)
    args = p.parse_args()
    result = compare_lm_head_pairwise_angles(args.ckpt_a, args.ckpt_b, meta=args.meta, device=args.device, lm_head_key_a=args.lm_head_key_a, lm_head_key_b=args.lm_head_key_b, min_angle=args.min_angle, max_angle=args.max_angle)
    for k, v in result.metrics.items(): print(f"{k}: {v:.6g}")
    if args.csv: write_csv(result, args.csv); print(f"wrote CSV: {args.csv}")
    if args.html: write_html(result, args.html); print(f"wrote HTML: {args.html}")

if __name__ == "__main__":
    main()
