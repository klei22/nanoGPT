# huggingface_model/gemma/270M/jl_head_eval.py
"""Evaluate JL-projected LM head top-n selection for Gemma 270M."""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Sequence

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, load_dataset_builder
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalConfig:
    model_name: str
    dataset_split: str
    fineweb_subset: str | None
    eval_tokens: int
    top_n_values: Sequence[int]
    target_dimensions: Sequence[int]
    seed: int
    projection: str
    device: str
    output_dir: str


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_fineweb_tokens(
    tokenizer: AutoTokenizer, dataset_split: str, fineweb_subset: str | None, max_tokens: int
) -> torch.Tensor:
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb", fineweb_subset or None, split=dataset_split)
    except ValueError as err:
        if "BuilderConfig" in str(err):
            builder = load_dataset_builder("HuggingFaceFW/fineweb")
            available = ", ".join(sorted(builder.builder_configs))
            raise ValueError(
                "Unknown FineWeb subset '{subset_name}'. Available configurations: {options}".format(
                    subset_name=fineweb_subset,
                    options=available,
                )
            ) from err
        raise

    tokens: List[int] = []
    target_len = max_tokens + 1
    for example in dataset:
        ids = tokenizer(example["text"])["input_ids"]
        tokens.extend(ids)
        if len(tokens) >= target_len:
            break

    if len(tokens) < target_len:
        raise ValueError(
            f"Need at least {target_len} tokens for evaluation, but only found {len(tokens)} tokens."
        )

    return torch.tensor(tokens[:target_len], dtype=torch.long).unsqueeze(0)


def _build_projection_matrix(
    hidden_dim: int, target_dim: int, seed: int, projection: str, device: str
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if projection == "gaussian":
        matrix = torch.randn((target_dim, hidden_dim), generator=generator)
    elif projection == "achlioptas":
        probs = torch.rand((target_dim, hidden_dim), generator=generator)
        matrix = torch.zeros((target_dim, hidden_dim))
        matrix[probs < 1 / 6] = 1.0
        matrix[(probs >= 1 / 6) & (probs < 2 / 6)] = -1.0
    else:
        raise ValueError(f"Unknown projection type: {projection}")

    matrix = matrix / math.sqrt(target_dim)
    return matrix.to(device)


def _ensure_labels_in_topk(topk: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels_expanded = labels.unsqueeze(-1)
    in_topk = (topk == labels_expanded).any(dim=-1)
    if in_topk.all():
        return topk
    updated = topk.clone()
    updated[~in_topk, -1] = labels[~in_topk]
    return updated


def _compute_topn_loss(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    approx_logits: torch.Tensor,
    top_n: int,
    labels: torch.Tensor,
) -> torch.Tensor:
    topk = approx_logits.topk(k=top_n, dim=-1).indices
    topk = _ensure_labels_in_topk(topk, labels)

    batch, seq_len, hidden_dim = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, hidden_dim)
    flat_indices = topk.reshape(-1, top_n)
    gathered_weight = weight[flat_indices]
    exact_logits = torch.einsum("bkh,bh->bk", gathered_weight, flat_hidden)
    exact_logits = exact_logits.view(batch, seq_len, top_n)

    flat_labels = labels.reshape(-1)
    blended_logits = approx_logits.clone()
    blended_logits = blended_logits.scatter(-1, topk, exact_logits)
    log_probs = torch.log_softmax(blended_logits, dim=-1)
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def _evaluate_grid(config: EvalConfig) -> List[List[float]]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name, attn_implementation="eager")
    model.to(config.device)
    model.eval()

    input_ids = _load_fineweb_tokens(
        tokenizer, config.dataset_split, config.fineweb_subset, config.eval_tokens
    ).to(config.device)
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[-1]

    vocab_size, hidden_dim = model.lm_head.weight.shape
    results: List[List[float]] = []
    for target_dim in config.target_dimensions:
        if target_dim <= 0 or target_dim > hidden_dim:
            raise ValueError(f"target_dimension must be in (0, {hidden_dim}], got {target_dim}")
        projection = _build_projection_matrix(
            hidden_dim=hidden_dim,
            target_dim=target_dim,
            seed=config.seed + target_dim,
            projection=config.projection,
            device=config.device,
        )
        with torch.no_grad():
            projected_hidden = torch.matmul(hidden_states, projection.T)
            projected_weight = torch.matmul(model.lm_head.weight, projection.T)
            approx_logits = torch.matmul(projected_hidden, projected_weight.T)
        row_results: List[float] = []
        for top_n in config.top_n_values:
            if top_n <= 0 or top_n > vocab_size:
                raise ValueError(f"top_n must be in (0, {vocab_size}], got {top_n}")
            with torch.no_grad():
                loss = _compute_topn_loss(
                    hidden_states,
                    model.lm_head.weight,
                    approx_logits,
                    top_n,
                    labels,
                )
            row_results.append(loss.item())
            print(f"target_dim={target_dim} top_n={top_n} loss={loss.item():.4f}")
        results.append(row_results)
    return results


def _write_csv(
    output_path: str,
    target_dimensions: Sequence[int],
    top_n_values: Sequence[int],
    results: Sequence[Sequence[float]],
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_dimension", *top_n_values])
        for target_dim, row in zip(target_dimensions, results):
            writer.writerow([target_dim, *row])


def _plot_heatmap(
    output_path: str,
    target_dimensions: Sequence[int],
    top_n_values: Sequence[int],
    results: Sequence[Sequence[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(results, aspect="auto", origin="lower")
    ax.set_xticks(range(len(top_n_values)), labels=[str(v) for v in top_n_values])
    ax.set_yticks(range(len(target_dimensions)), labels=[str(v) for v in target_dimensions])
    ax.set_xlabel("Top-N candidates")
    ax.set_ylabel("Target dimension")
    ax.set_title("JL-projected LM head validation loss")
    fig.colorbar(image, ax=ax, label="Validation loss")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate JL-projected LM head top-n filtering.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m",
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train[:1%]",
        help="FineWeb dataset split to sample tokens from.",
    )
    parser.add_argument(
        "--fineweb_subset",
        type=str,
        default="sample-10BT",
        help="FineWeb subset configuration. Use empty string for the default config.",
    )
    parser.add_argument(
        "--eval_tokens",
        type=int,
        default=1000,
        help="Number of tokens to evaluate (uses eval_tokens + 1 for the shift).",
    )
    parser.add_argument(
        "--top_n_values",
        type=_parse_int_list,
        default=_parse_int_list("1000,2000,5000,10000"),
        help="Comma-separated list of top-n vocabulary sizes to evaluate.",
    )
    parser.add_argument(
        "--target_dimensions",
        type=_parse_int_list,
        default=_parse_int_list("500,400,300,200,100"),
        help="Comma-separated list of JL target dimensions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for JL projections.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        choices=("gaussian", "achlioptas"),
        default="gaussian",
        help="JL projection matrix type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="jl_eval_outputs",
        help="Directory to store CSV and heatmap outputs.",
    )
    args = parser.parse_args()
    fineweb_subset = args.fineweb_subset or None
    return EvalConfig(
        model_name=args.model_name,
        dataset_split=args.dataset_split,
        fineweb_subset=fineweb_subset,
        eval_tokens=args.eval_tokens,
        top_n_values=args.top_n_values,
        target_dimensions=args.target_dimensions,
        seed=args.seed,
        projection=args.projection,
        device=args.device,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = _parse_args()
    os.makedirs(config.output_dir, exist_ok=True)
    results = _evaluate_grid(config)

    csv_path = os.path.join(config.output_dir, "jl_head_eval_results.csv")
    heatmap_path = os.path.join(config.output_dir, "jl_head_eval_heatmap.png")
    _write_csv(csv_path, config.target_dimensions, config.top_n_values, results)
    _plot_heatmap(heatmap_path, config.target_dimensions, config.top_n_values, results)
    print(f"Saved results to {csv_path} and {heatmap_path}")


if __name__ == "__main__":
    main()
