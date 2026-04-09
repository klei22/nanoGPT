#!/usr/bin/env python3
"""Compare immediate next-token distributions for multiple checkpoints.

This script runs first-association analysis (single-token context) over a set of
start tokens and compares model outputs with user-provided labels.
"""

import argparse
import json
import os
import pickle
import re
import sys
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare immediate logit/probability outputs for multiple checkpoints")
    p.add_argument("--model_out_dir", nargs='+', required=True, help="Checkpoint directories (each contains ckpt.pt)")
    p.add_argument("--model_label", nargs='+', required=True, help="Display labels for models (e.g., embd256 embd384 embd512)")
    p.add_argument(
        "--input_tokens_yaml",
        default="all",
        type=str,
        help="YAML file describing start tokens to sweep, or 'all' to use full vocab.",
    )
    p.add_argument("--output_dir", required=True, type=str, help="Where plots and summary artifacts are written")
    p.add_argument("--top_k", type=int, default=20, help="Top-k logits to use for histogram plots")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for token sweeps")
    p.add_argument("--device", type=str, default="cuda", help="Device for inference")
    p.add_argument("--weights_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--reference_label", type=str, default=None, help="Reference label used for KL bar chart; defaults to first label")
    p.add_argument(
        "--save_logits_yaml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-model probability outputs for reuse.",
    )
    return p.parse_args()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "model"


def _load_model(out_dir: str, device: str, weights_only: bool) -> tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    load_kwargs = {"map_location": device}
    if "weights_only" in signature(torch.load).parameters:
        load_kwargs["weights_only"] = weights_only
    checkpoint = torch.load(ckpt_path, **load_kwargs)
    checkpoint["model_args"]["dropout"] = 0.0

    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, checkpoint


def _load_meta(out_dir: str, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    dataset = checkpoint.get("config", {}).get("dataset")
    candidates = [
        os.path.join(out_dir, "meta.pkl"),
        os.path.join("data", dataset, "meta.pkl") if dataset else None,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not find meta.pkl for checkpoint in {out_dir}")


def _extract_token_ids_from_yaml(path: str, vocab_size: int) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        token_ids = data
    elif isinstance(data, dict):
        for key in ("start_tokens", "token_ids", "tokens", "input_tokens"):
            if key in data:
                token_ids = data[key]
                break
        else:
            token_ids = [int(k) for k in data.keys()]
    else:
        raise ValueError("Unsupported YAML format for input tokens")

    cleaned = sorted({int(t) for t in token_ids})
    for tid in cleaned:
        if tid < 0 or tid >= vocab_size:
            raise ValueError(f"Token id {tid} outside vocab range [0, {vocab_size-1}]")
    return cleaned


def _resolve_start_tokens(input_tokens_yaml: str, vocab_size: int) -> List[int]:
    if input_tokens_yaml.lower() == "all":
        return list(range(vocab_size))
    return _extract_token_ids_from_yaml(input_tokens_yaml, vocab_size)


def _token_label(meta: Dict[str, Any], token_id: int) -> str:
    itos = meta.get("itos")
    if isinstance(itos, dict):
        value = itos.get(token_id)
    elif isinstance(itos, list) and token_id < len(itos):
        value = itos[token_id]
    else:
        value = None

    if value is None:
        return str(token_id)

    text = str(value).replace("\n", "\\n").replace("\t", "\\t")
    return f"{token_id}:{text}"


def _extract_vocab_labels(meta: Dict[str, Any], vocab_size: int) -> Dict[int, str]:
    itos = meta.get("itos")
    labels: Dict[int, str] = {}
    if isinstance(itos, dict):
        for i in range(vocab_size):
            value = itos.get(i)
            if value is not None:
                labels[i] = str(value)
    elif isinstance(itos, list):
        for i, value in enumerate(itos[:vocab_size]):
            labels[i] = str(value)
    return labels


def _sweep_logits(model: torch.nn.Module, token_ids: Sequence[int], device: str, batch_size: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(token_ids), batch_size):
            chunk = token_ids[i : i + batch_size]
            x = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(1)
            logits, _ = model(x, dataset_idx=0)
            rows.append(logits[:, -1, :].to(torch.float32).cpu())
    return torch.cat(rows, dim=0)


def _save_topk_hist_by_model(logits_by_label: Dict[str, torch.Tensor], top_k: int, out_path: str) -> None:
    labels = list(logits_by_label.keys())
    cols = min(3, len(labels))
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    for i, label in enumerate(labels):
        ax = axes[i // cols][i % cols]
        logits = logits_by_label[label]
        k = min(top_k, logits.size(-1))
        values = torch.topk(logits, k=k, dim=-1).values.reshape(-1).numpy()
        ax.hist(values, bins=100, alpha=0.85)
        ax.set_title(f"{label} top-{k} logits")
        ax.set_xlabel("Logit")
        ax.set_ylabel("Count")

    for j in range(len(labels), rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _save_kl_barh(
    kl_by_label: Dict[str, np.ndarray],
    labels: Sequence[str],
    out_path: str,
    reference_label: str,
) -> None:
    models = list(kl_by_label.keys())
    n_models = len(models)
    y = np.arange(len(labels), dtype=np.float64)
    bar_h = 0.8 / max(n_models, 1)

    height = max(6.0, len(labels) * 0.16)
    fig, ax = plt.subplots(figsize=(15, height))

    for m_idx, model_label in enumerate(models):
        offset = (m_idx - (n_models - 1) / 2.0) * bar_h
        ax.barh(y + offset, kl_by_label[model_label], height=bar_h * 0.95, alpha=0.85, label=model_label)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"KL divergence KL({reference_label} || model)")
    ax.set_ylabel("Start token")
    ax.set_title("Per-start-token alignment (total KL over full vocabulary)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _save_probs_yaml(
    path: Path,
    label: str,
    token_ids: Sequence[int],
    probs: torch.Tensor,
    vocab_labels: Dict[int, str],
) -> None:
    payload = {
        "label": label,
        "start_tokens": [int(t) for t in token_ids],
        "vocab_labels": {int(k): v for k, v in vocab_labels.items()},
        "probabilities": probs.tolist(),
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    if len(args.model_out_dir) != len(args.model_label):
        raise ValueError("--model_out_dir and --model_label must have the same number of entries")
    if len(set(args.model_label)) != len(args.model_label):
        raise ValueError("--model_label values must be unique")
    if len(args.model_out_dir) < 2:
        raise ValueError("Provide at least two models")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_labels = list(args.model_label)
    reference_label = args.reference_label or model_labels[0]
    if reference_label not in model_labels:
        raise ValueError(f"reference_label '{reference_label}' not found in model_label list")

    models: Dict[str, torch.nn.Module] = {}
    metas: Dict[str, Dict[str, Any]] = {}
    vocabs: Dict[str, int] = {}

    for label, out_dir in zip(model_labels, args.model_out_dir):
        model, checkpoint = _load_model(out_dir, args.device, args.weights_only)
        meta = _load_meta(out_dir, checkpoint)
        vocab = int(meta.get("vocab_size", model.config.vocab_size))
        models[label] = model
        metas[label] = meta
        vocabs[label] = vocab

    vocab_sizes = set(vocabs.values())
    if len(vocab_sizes) != 1:
        raise ValueError(f"All models must share vocab size, got: {vocabs}")
    vocab_size = next(iter(vocab_sizes))

    token_ids = _resolve_start_tokens(args.input_tokens_yaml, vocab_size)
    token_labels = [_token_label(metas[reference_label], t) for t in token_ids]

    logits_by_label: Dict[str, torch.Tensor] = {}
    probs_by_label: Dict[str, torch.Tensor] = {}

    for label in model_labels:
        logits = _sweep_logits(models[label], token_ids, args.device, args.batch_size)
        probs = F.softmax(logits, dim=-1)
        logits_by_label[label] = logits
        probs_by_label[label] = probs

    _save_topk_hist_by_model(
        logits_by_label,
        args.top_k,
        str(output_dir / "topk_logit_hist_by_model.png"),
    )

    ref_probs = probs_by_label[reference_label]
    kl_by_label: Dict[str, np.ndarray] = {}
    for label in model_labels:
        if label == reference_label:
            continue
        cur_probs = probs_by_label[label]
        kl_vec = F.kl_div(torch.log(ref_probs + 1e-12), cur_probs, reduction="none").sum(dim=-1).numpy()
        kl_by_label[label] = kl_vec
        np.save(output_dir / f"per_token_kl_{_slugify(reference_label)}_vs_{_slugify(label)}.npy", kl_vec)

    _save_kl_barh(
        kl_by_label,
        token_labels,
        str(output_dir / "per_token_kl_barh.png"),
        reference_label,
    )

    pairwise_stats: Dict[str, Dict[str, float]] = {}
    for label, kl_vec in kl_by_label.items():
        pairwise_stats[f"{reference_label}_vs_{label}"] = {
            "kl_mean": float(np.mean(kl_vec)),
            "kl_std": float(np.std(kl_vec)),
            "kl_min": float(np.min(kl_vec)),
            "kl_max": float(np.max(kl_vec)),
        }

    summary = {
        "model_labels": model_labels,
        "model_out_dirs": {label: out_dir for label, out_dir in zip(model_labels, args.model_out_dir)},
        "reference_label": reference_label,
        "num_start_tokens": len(token_ids),
        "top_k_for_hist": int(min(args.top_k, vocab_size)),
        "vocab_size": vocab_size,
        "pairwise_reference_kl_stats": pairwise_stats,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.save_logits_yaml:
        manifest: Dict[str, str] = {}
        for label in model_labels:
            yaml_name = f"probs_{_slugify(label)}.yaml"
            manifest[label] = yaml_name
            _save_probs_yaml(
                output_dir / yaml_name,
                label,
                token_ids,
                probs_by_label[label],
                _extract_vocab_labels(metas[label], vocab_size),
            )

        with (output_dir / "probs_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"Wrote analysis artifacts to {output_dir}")


if __name__ == "__main__":
    main()
