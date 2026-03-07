#!/usr/bin/env python3
"""Build an interactive Plotly page for first-association top-k comparisons.

Input files are the YAML probability dumps produced by compare_first_association.py.
For each start token, this script renders side-by-side horizontal bar charts for model A
and model B over the union of each model's top-k predicted next tokens.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive top-k next-token comparison from model probability YAMLs")
    p.add_argument("--model_a_probs_yaml", required=True, type=Path)
    p.add_argument("--model_b_probs_yaml", required=True, type=Path)
    p.add_argument("--output_html", required=True, type=Path, help="Output interactive Plotly HTML")
    p.add_argument("--top_k", type=int, default=20, help="Top-k token candidates per model to compare")
    p.add_argument("--output_json", type=Path, default=None, help="Optional summary JSON output")
    return p.parse_args()


def _load_prob_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {path}: expected mapping")
    if "start_tokens" not in data or "probabilities" not in data:
        raise ValueError(f"{path} must contain 'start_tokens' and 'probabilities'")
    return data


def _decode_label(vocab_labels: Dict[int, str], token_id: int) -> str:
    text = vocab_labels.get(int(token_id))
    if text is None:
        return str(token_id)
    escaped = str(text).replace("\n", "\\n").replace("\t", "\\t")
    return f"{token_id}:{escaped}"


def _to_vocab_label_map(data: Dict[str, Any]) -> Dict[int, str]:
    raw = data.get("vocab_labels", {})
    out: Dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[int(k)] = str(v)
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            out[i] = str(v)
    return out


def _validate_inputs(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[List[int], np.ndarray, np.ndarray, Dict[int, str], Dict[int, str]]:
    start_a = [int(x) for x in a["start_tokens"]]
    start_b = [int(x) for x in b["start_tokens"]]
    if start_a != start_b:
        raise ValueError("start_tokens do not match between model A and model B YAML files")

    probs_a = np.asarray(a["probabilities"], dtype=np.float64)
    probs_b = np.asarray(b["probabilities"], dtype=np.float64)
    if probs_a.shape != probs_b.shape:
        raise ValueError(f"Probability tensor shapes differ: {probs_a.shape} vs {probs_b.shape}")
    if probs_a.ndim != 2:
        raise ValueError(f"Expected probabilities with shape [num_start_tokens, vocab_size], got {probs_a.shape}")

    return start_a, probs_a, probs_b, _to_vocab_label_map(a), _to_vocab_label_map(b)


def _topk_union_indices(pa: np.ndarray, pb: np.ndarray, k: int) -> List[int]:
    k = max(1, min(k, pa.shape[0], pb.shape[0]))
    top_a = np.argpartition(pa, -k)[-k:]
    top_b = np.argpartition(pb, -k)[-k:]
    union = np.unique(np.concatenate([top_a, top_b]))
    # sort union by max probability across models, descending
    scores = np.maximum(pa[union], pb[union])
    order = np.argsort(-scores)
    return union[order].tolist()


def build_figure(
    start_tokens: Sequence[int],
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    vocab_a: Dict[int, str],
    vocab_b: Dict[int, str],
    top_k: int,
) -> go.Figure:
    fig = go.Figure()
    n = len(start_tokens)

    steps = []
    for i, start_token in enumerate(start_tokens):
        pa = probs_a[i]
        pb = probs_b[i]
        candidate_ids = _topk_union_indices(pa, pb, top_k)

        # use model A label when available, fallback to B
        y_labels = [
            _decode_label(vocab_a if cid in vocab_a else vocab_b, cid)
            for cid in candidate_ids
        ]
        x_a = [float(pa[cid]) for cid in candidate_ids]
        x_b = [float(pb[cid]) for cid in candidate_ids]

        visible = [False] * (2 * n)
        visible[2 * i] = True
        visible[2 * i + 1] = True

        fig.add_trace(
            go.Bar(
                x=x_a,
                y=y_labels,
                orientation="h",
                name="Model A",
                marker_color="#1f77b4",
                visible=(i == 0),
            )
        )
        fig.add_trace(
            go.Bar(
                x=x_b,
                y=y_labels,
                orientation="h",
                name="Model B",
                marker_color="#ff7f0e",
                visible=(i == 0),
            )
        )

        start_title = _decode_label(vocab_a if int(start_token) in vocab_a else vocab_b, int(start_token))
        steps.append(
            {
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Start token: {start_title}"},
                ],
                "label": str(i),
            }
        )

    first_title = _decode_label(vocab_a if int(start_tokens[0]) in vocab_a else vocab_b, int(start_tokens[0])) if n else ""
    fig.update_layout(
        title=f"Start token: {first_title}",
        barmode="group",
        xaxis_title="Next-token probability",
        yaxis_title="Next token (index:word)",
        height=900,
        margin=dict(l=260, r=30, t=60, b=80),
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Start-token index: "},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main() -> None:
    args = parse_args()
    a = _load_prob_yaml(args.model_a_probs_yaml)
    b = _load_prob_yaml(args.model_b_probs_yaml)

    start_tokens, probs_a, probs_b, vocab_a, vocab_b = _validate_inputs(a, b)
    fig = build_figure(start_tokens, probs_a, probs_b, vocab_a, vocab_b, args.top_k)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output_html), include_plotlyjs="cdn", full_html=True)

    if args.output_json is not None:
        summary = {
            "num_start_tokens": len(start_tokens),
            "vocab_size": int(probs_a.shape[1]),
            "top_k": int(args.top_k),
            "output_html": str(args.output_html),
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Wrote interactive plot to {args.output_html}")


if __name__ == "__main__":
    main()
