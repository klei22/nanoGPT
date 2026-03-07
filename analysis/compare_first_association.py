#!/usr/bin/env python3
"""Compare immediate next-token distributions for two checkpoints.

This script focuses on "first association" style analysis where each start token is
fed alone (single-token context), then logits/probabilities are compared between two
models for the same start-token set.
"""

import argparse
import json
import os
import pickle
from inspect import signature
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare immediate logit/probability outputs for two checkpoints")
    p.add_argument("--model_a_out_dir", required=True, type=str, help="Checkpoint directory for model A (contains ckpt.pt)")
    p.add_argument("--model_b_out_dir", required=True, type=str, help="Checkpoint directory for model B (contains ckpt.pt)")
    p.add_argument(
        "--input_tokens_yaml",
        default="all",
        type=str,
        help="YAML file describing start tokens to sweep, or 'all' to use full vocab.",
    )
    p.add_argument("--output_dir", required=True, type=str, help="Where plots and summary artifacts are written")
    p.add_argument("--top_k", type=int, default=20, help="Top-k logits to use for side-by-side histogram plots")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for token sweeps")
    p.add_argument("--device", type=str, default="cuda", help="Device for inference")
    p.add_argument("--weights_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--save_logits_yaml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-token probability outputs for reuse.",
    )
    return p.parse_args()


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
            # fallback: if mapping by token id -> metadata, use keys
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


def _sweep_logits(model: torch.nn.Module, token_ids: Sequence[int], device: str, batch_size: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(token_ids), batch_size):
            chunk = token_ids[i : i + batch_size]
            x = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(1)
            logits, _ = model(x, dataset_idx=0)
            rows.append(logits[:, -1, :].to(torch.float32).cpu())
    return torch.cat(rows, dim=0)


def _save_topk_side_by_side(logits_a: torch.Tensor, logits_b: torch.Tensor, top_k: int, out_path: str) -> None:
    k = min(top_k, logits_a.size(-1), logits_b.size(-1))
    a_vals = torch.topk(logits_a, k=k, dim=-1).values.reshape(-1).numpy()
    b_vals = torch.topk(logits_b, k=k, dim=-1).values.reshape(-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes[0].hist(a_vals, bins=100, color="#1f77b4", alpha=0.85)
    axes[0].set_title(f"Model A top-{k} logits")
    axes[0].set_xlabel("Logit")
    axes[0].set_ylabel("Count")

    axes[1].hist(b_vals, bins=100, color="#ff7f0e", alpha=0.85)
    axes[1].set_title(f"Model B top-{k} logits")
    axes[1].set_xlabel("Logit")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _save_kl_barh(kl_values: np.ndarray, labels: Sequence[str], out_path: str) -> None:
    height = max(6.0, len(labels) * 0.16)
    fig, ax = plt.subplots(figsize=(14, height))
    y = np.arange(len(labels))
    ax.barh(y, kl_values, color="#2ca02c", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("KL divergence KL(model_a || model_b)")
    ax.set_ylabel("Start token")
    ax.set_title("Per-start-token alignment (total KL over full vocabulary)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


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


def _save_probs_yaml(path: str, token_ids: Sequence[int], probs: torch.Tensor, vocab_labels: Dict[int, str]) -> None:
    payload = {
        "start_tokens": [int(t) for t in token_ids],
        "vocab_labels": {int(k): v for k, v in vocab_labels.items()},
        "probabilities": probs.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_a, ckpt_a = _load_model(args.model_a_out_dir, args.device, args.weights_only)
    model_b, ckpt_b = _load_model(args.model_b_out_dir, args.device, args.weights_only)

    meta_a = _load_meta(args.model_a_out_dir, ckpt_a)
    meta_b = _load_meta(args.model_b_out_dir, ckpt_b)

    vocab_a = int(meta_a.get("vocab_size", model_a.config.vocab_size))
    vocab_b = int(meta_b.get("vocab_size", model_b.config.vocab_size))
    if vocab_a != vocab_b:
        raise ValueError(f"Mismatched vocab sizes: model_a={vocab_a}, model_b={vocab_b}")

    token_ids = _resolve_start_tokens(args.input_tokens_yaml, vocab_a)

    logits_a = _sweep_logits(model_a, token_ids, args.device, args.batch_size)
    logits_b = _sweep_logits(model_b, token_ids, args.device, args.batch_size)

    probs_a = F.softmax(logits_a, dim=-1)
    probs_b = F.softmax(logits_b, dim=-1)

    kl_values = F.kl_div(torch.log(probs_a + 1e-12), probs_b, reduction="none").sum(dim=-1).numpy()
    labels = [_token_label(meta_a, t) for t in token_ids]

    hist_path = os.path.join(args.output_dir, "topk_logit_hist_side_by_side.png")
    _save_topk_side_by_side(logits_a, logits_b, args.top_k, hist_path)

    barh_path = os.path.join(args.output_dir, "per_token_kl_barh.png")
    _save_kl_barh(kl_values, labels, barh_path)

    ranking = sorted(zip(token_ids, labels, kl_values.tolist()), key=lambda x: x[2], reverse=True)
    summary = {
        "model_a_out_dir": args.model_a_out_dir,
        "model_b_out_dir": args.model_b_out_dir,
        "num_start_tokens": len(token_ids),
        "top_k_for_hist": int(min(args.top_k, vocab_a)),
        "vocab_size": vocab_a,
        "kl_mean": float(np.mean(kl_values)),
        "kl_std": float(np.std(kl_values)),
        "kl_min": float(np.min(kl_values)),
        "kl_max": float(np.max(kl_values)),
        "top_10_highest_kl_tokens": [
            {"token_id": int(t), "label": lbl, "kl": float(kl)}
            for t, lbl, kl in ranking[:10]
        ],
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(args.output_dir, "per_token_kl.npy"), kl_values)

    if args.save_logits_yaml:
        vocab_labels_a = _extract_vocab_labels(meta_a, vocab_a)
        vocab_labels_b = _extract_vocab_labels(meta_b, vocab_b)
        _save_probs_yaml(os.path.join(args.output_dir, "model_a_probs.yaml"), token_ids, probs_a, vocab_labels_a)
        _save_probs_yaml(os.path.join(args.output_dir, "model_b_probs.yaml"), token_ids, probs_b, vocab_labels_b)

    print(f"Wrote analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
