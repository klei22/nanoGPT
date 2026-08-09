#!/usr/bin/env python3
"""Compare immediate next-token logit histograms between two GPT configs.

This script sweeps through every possible input token id for a shared vocabulary,
collects each model's immediate next-token logits, and then:
  * Saves top-k logit histogram bar charts for model A and model B
  * Saves a histogram of per-token KL(A || B) divergences
  * Writes a JSON summary with aggregate stats and top-k token details
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from sample import get_tokenizer_functions, plot_topk_logit_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Immediate histogram + KL demo")
    parser.add_argument("--dataset", default="shakespeare_char", help="Dataset name under data/")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--out_dir", default="out/immediate_histogram_kl_demo", help="Output directory")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--top_k", type=int, default=20, help="Top-k tokens to visualize in logit histograms")
    parser.add_argument("--bins", type=int, default=40, help="Bins for KL histogram")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for KL computation")
    return parser.parse_args()


def load_dataset_meta(dataset: str) -> Dict:
    meta_path = Path("data") / dataset / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing tokenizer metadata: {meta_path}")
    with meta_path.open("rb") as f:
        return pickle.load(f)


def build_demo_config(vocab_size: int, n_embd: int) -> GPTConfig:
    # Based on the default_inf.yaml-style choices:
    # qk_norm + peri_ln + rotary + infinite + mqa + concat heads
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=6,
        n_head=3,
        n_kv_group=1,
        n_embd=n_embd,
        use_qk_norm=True,
        use_qk_norm_scale=True,
        use_pre_ln=True,
        use_peri_ln=True,
        use_post_ln=False,
        use_rotary_embeddings=True,
        use_abs_pos_embeddings=False,
        attention_variant="infinite",
        use_concat_heads=True,
        n_qk_head_dim=100,
        n_v_head_dim=100,
        softmax_variant_attn="softmax",
        dropout=0.0,
    )


@torch.no_grad()
def sweep_immediate_logits(model: GPT, vocab_size: int, device: str) -> torch.Tensor:
    model.eval()
    token_ids = torch.arange(vocab_size, dtype=torch.long, device=device).unsqueeze(1)
    logits, _ = model(token_ids)
    return logits[:, -1, :].float().cpu()


def compute_kl_per_input(logits_a: torch.Tensor, logits_b: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_a = logits_a / temperature
    scaled_b = logits_b / temperature
    log_p_a = F.log_softmax(scaled_a, dim=-1)
    log_p_b = F.log_softmax(scaled_b, dim=-1)
    p_a = log_p_a.exp()
    return (p_a * (log_p_a - log_p_b)).sum(dim=-1)


def save_kl_histogram(kl_values: torch.Tensor, out_path: Path, bins: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(kl_values.numpy(), bins=bins)
    plt.xlabel("KL(A || B) per input token")
    plt.ylabel("Count")
    plt.title("Distribution of Immediate KL Divergences")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize(kl_values: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(kl_values.mean().item()),
        "std": float(kl_values.std(unbiased=False).item()),
        "min": float(kl_values.min().item()),
        "max": float(kl_values.max().item()),
        "median": float(kl_values.median().item()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_dataset_meta(args.dataset)
    tokenizer_name = meta.get("tokenizer", "char")
    if tokenizer_name not in {"char", "char_with_byte_fallback"}:
        raise ValueError(f"Expected char-based tokenizer, found: {tokenizer_name}")

    _, decode = get_tokenizer_functions(meta)
    vocab_size = int(meta["vocab_size"])

    cfg_a = build_demo_config(vocab_size=vocab_size, n_embd=384)
    cfg_b = build_demo_config(vocab_size=vocab_size, n_embd=320)

    model_a = GPT(cfg_a).to(args.device)
    model_b = GPT(cfg_b).to(args.device)

    logits_a = sweep_immediate_logits(model_a, vocab_size, args.device)
    logits_b = sweep_immediate_logits(model_b, vocab_size, args.device)

    kl_values = compute_kl_per_input(logits_a, logits_b, temperature=args.temperature)

    mean_logits_a = logits_a.mean(dim=0)
    mean_logits_b = logits_b.mean(dim=0)

    model_a_chart = out_dir / "model_a_topk_logit_histogram.png"
    model_b_chart = out_dir / "model_b_topk_logit_histogram.png"
    kl_chart = out_dir / "kl_divergence_distribution.png"

    top_a = plot_topk_logit_histogram(
        logits=mean_logits_a,
        decode=decode,
        top_k=args.top_k,
        title=f"Model A (n_embd=384) mean logits: top-{args.top_k}",
        out_path=str(model_a_chart),
    )
    top_b = plot_topk_logit_histogram(
        logits=mean_logits_b,
        decode=decode,
        top_k=args.top_k,
        title=f"Model B (n_embd=320) mean logits: top-{args.top_k}",
        out_path=str(model_b_chart),
    )
    save_kl_histogram(kl_values, kl_chart, bins=args.bins)

    summary = {
        "dataset": args.dataset,
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "config_a": {"n_embd": 384, "n_head": 3, "n_qk_head_dim": 100, "n_v_head_dim": 100, "use_concat_heads": True},
        "config_b": {"n_embd": 320, "n_head": 3, "n_qk_head_dim": 100, "n_v_head_dim": 100, "use_concat_heads": True},
        "kl_stats": summarize(kl_values),
        "artifacts": {
            "model_a_topk_histogram": str(model_a_chart),
            "model_b_topk_histogram": str(model_b_chart),
            "kl_distribution": str(kl_chart),
        },
        "top_tokens_model_a": top_a,
        "top_tokens_model_b": top_b,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
