#!/usr/bin/env python3
"""Build an interactive Plotly page for first-association top-k comparisons.

Input files are per-model YAML probability dumps produced by
analysis/compare_first_association.py.
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
    p.add_argument("--probs_yaml", nargs='+', required=True, help="Paths to model probability YAML files")
    p.add_argument("--label", nargs='+', required=True, help="Labels corresponding to probs_yaml inputs")
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
    loaded: List[Dict[str, Any]],
) -> Tuple[List[int], List[np.ndarray], List[Dict[int, str]]]:
    start_ref = [int(x) for x in loaded[0]["start_tokens"]]
    probs_list: List[np.ndarray] = []
    vocab_maps: List[Dict[int, str]] = []
    shape_ref: Tuple[int, ...] | None = None

    for data in loaded:
        start = [int(x) for x in data["start_tokens"]]
        if start != start_ref:
            raise ValueError("start_tokens do not match across provided YAML files")

        probs = np.asarray(data["probabilities"], dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"Expected probabilities [num_start_tokens, vocab_size], got {probs.shape}")

        if shape_ref is None:
            shape_ref = probs.shape
        elif probs.shape != shape_ref:
            raise ValueError(f"Probability tensor shapes differ: expected {shape_ref}, got {probs.shape}")

        probs_list.append(probs)
        vocab_maps.append(_to_vocab_label_map(data))

    return start_ref, probs_list, vocab_maps


def _topk_union_indices(prob_rows: Sequence[np.ndarray], k: int) -> List[int]:
    vocab_size = int(prob_rows[0].shape[0])
    k = max(1, min(k, vocab_size))
    parts = []
    for row in prob_rows:
        parts.append(np.argpartition(row, -k)[-k:])
    union = np.unique(np.concatenate(parts))
    max_scores = np.max(np.stack([row[union] for row in prob_rows], axis=0), axis=0)
    order = np.argsort(-max_scores)
    return union[order].tolist()


def build_figure(
    start_tokens: Sequence[int],
    probs_list: Sequence[np.ndarray],
    vocab_maps: Sequence[Dict[int, str]],
    labels: Sequence[str],
    top_k: int,
) -> go.Figure:
    fig = go.Figure()
    n_tokens = len(start_tokens)
    n_models = len(labels)

    steps = []
    for i, start_token in enumerate(start_tokens):
        per_model_rows = [p[i] for p in probs_list]
        candidate_ids = _topk_union_indices(per_model_rows, top_k)

        y_labels = []
        for cid in candidate_ids:
            decoded = None
            for vocab in vocab_maps:
                if int(cid) in vocab:
                    decoded = _decode_label(vocab, int(cid))
                    break
            y_labels.append(decoded if decoded is not None else str(int(cid)))

        visible = [False] * (n_tokens * n_models)
        for m_idx in range(n_models):
            visible[i * n_models + m_idx] = True

        for m_idx, label in enumerate(labels):
            x_vals = [float(per_model_rows[m_idx][cid]) for cid in candidate_ids]
            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_labels,
                    orientation="h",
                    name=label,
                    visible=(i == 0),
                )
            )

        title_label = None
        for vocab in vocab_maps:
            if int(start_token) in vocab:
                title_label = _decode_label(vocab, int(start_token))
                break
        if title_label is None:
            title_label = str(int(start_token))

        steps.append(
            {
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Start token: {title_label}"},
                ],
                "label": str(i),
            }
        )

    first_title = str(int(start_tokens[0])) if start_tokens else ""
    if start_tokens:
        for vocab in vocab_maps:
            if int(start_tokens[0]) in vocab:
                first_title = _decode_label(vocab, int(start_tokens[0]))
                break

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
    if len(args.probs_yaml) != len(args.label):
        raise ValueError("--probs_yaml and --label must have the same number of entries")
    if len(set(args.label)) != len(args.label):
        raise ValueError("--label entries must be unique")
    if len(args.probs_yaml) < 2:
        raise ValueError("Provide at least two probability YAML files")

    loaded = [_load_prob_yaml(Path(p)) for p in args.probs_yaml]
    start_tokens, probs_list, vocab_maps = _validate_inputs(loaded)

    fig = build_figure(start_tokens, probs_list, vocab_maps, args.label, args.top_k)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output_html), include_plotlyjs="cdn", full_html=True)

    if args.output_json is not None:
        summary = {
            "num_start_tokens": len(start_tokens),
            "vocab_size": int(probs_list[0].shape[1]),
            "top_k": int(args.top_k),
            "labels": list(args.label),
            "inputs": list(args.probs_yaml),
            "output_html": str(args.output_html),
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Wrote interactive plot to {args.output_html}")


if __name__ == "__main__":
    main()
