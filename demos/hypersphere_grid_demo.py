#!/usr/bin/env python3
"""Demo script for hypersphere grid projection on transformer pre-norm activations."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import GPT, GPTConfig
from sample import (
    calculate_validation_loss,
    generate_random_hypersphere_grid,
    load_validation_data,
    prepare_hypersphere_grid_tensor,
    set_hypersphere_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hypersphere grid projection demo")
    parser.add_argument("--out_dir", type=str, default="out",
                        help="Directory containing the training checkpoint (ckpt.pt)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for evaluation (e.g., 'cpu', 'cuda', 'cuda:0')")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Computation dtype for autocast during evaluation")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Name of dataset to use for validation loss (defaults to checkpoint dataset)")
    parser.add_argument("--eval_iters", type=int, default=50,
                        help="Number of evaluation iterations per grid size")
    parser.add_argument("--vector_counts", type=int, nargs="+",
                        default=[10_000, 100_000, 1_000_000],
                        help="List of hypersphere grid sizes to evaluate")
    parser.add_argument("--grid_method", type=str, default="random",
                        choices=["random"],
                        help="Grid generation method (currently only 'random')")
    parser.add_argument("--grid_mean", type=float, default=0.0,
                        help="Mean for randomly generated grid vectors")
    parser.add_argument("--grid_std", type=float, default=0.02,
                        help="Stddev for randomly generated grid vectors")
    parser.add_argument("--grid_normalize", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="L2-normalize generated grid vectors before use")
    parser.add_argument("--grid_path", type=str, default=None,
                        help="Optional NumPy file containing a precomputed grid to reuse")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducibility")
    parser.add_argument("--plot_path", type=Path, default=Path("hypersphere_grid_eval.png"),
                        help="File path to save the validation loss plot")
    parser.add_argument("--metrics_path", type=Path, default=Path("hypersphere_grid_eval.json"),
                        help="File path to save validation metrics as JSON")
    return parser.parse_args()


def load_model_from_checkpoint(out_dir: str, device: str) -> Tuple[GPT, dict]:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint["model_args"]["dropout"] = 0.0
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    return model, checkpoint


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    ptdtype = dtype_map[args.dtype]

    model, checkpoint = load_model_from_checkpoint(args.out_dir, args.device)
    model.eval()
    model.to(args.device)

    dataset_name = args.eval_dataset
    if dataset_name is None:
        dataset_name = checkpoint.get("config", {}).get("dataset")
    if dataset_name is None:
        raise ValueError("Evaluation dataset must be specified via --eval_dataset or checkpoint metadata")

    val_data = load_validation_data(model.config.block_size, dataset_name)

    vector_counts = list(args.vector_counts)
    results: List[dict] = []
    losses: List[float] = []

    loaded_grid: Optional[torch.Tensor] = None
    if args.grid_path:
        loaded_array = np.load(args.grid_path)
        loaded_grid = prepare_hypersphere_grid_tensor(
            loaded_array,
            model.config.n_embd,
        )

    for idx, count in enumerate(vector_counts):
        if count <= 0:
            raise ValueError("Vector counts must be positive integers")

        print(f"[demo] Evaluating hypersphere grid with {count:,} vectors...")
        if loaded_grid is not None:
            if loaded_grid.shape[0] < count:
                raise ValueError(
                    "Provided grid file contains fewer vectors than required: "
                    f"needed {count}, found {loaded_grid.shape[0]}"
                )
            grid = loaded_grid[:count].contiguous()
        else:
            if args.grid_method != "random":
                raise ValueError(f"Unsupported grid generation method: {args.grid_method}")
            # advance generator state for each iteration for reproducibility
            local_generator = torch.Generator(device="cpu")
            local_generator.manual_seed(args.seed + idx)
            grid = generate_random_hypersphere_grid(
                count,
                model.config.n_embd,
                mean=args.grid_mean,
                std=args.grid_std,
                generator=local_generator,
            )

        set_hypersphere_grid(model, grid, normalize=args.grid_normalize)

        metrics = calculate_validation_loss(
            model,
            val_data,
            model.config.block_size,
            args.eval_iters,
            args.device,
            ptdtype,
        )
        val_loss = metrics.get("val", float("nan"))
        losses.append(val_loss)
        results.append({
            "num_vectors": int(count),
            "val": val_loss,
            "val_std": metrics.get("val_std"),
            "elapsed_time_s": metrics.get("elapsed_time_s"),
        })
        print(f"[demo] Validation loss with {count:,} vectors: {val_loss:.4f}")

    # Clean up grid hooks
    set_hypersphere_grid(model, None)

    plt.figure(figsize=(8, 5))
    plt.plot(vector_counts, losses, marker="o", linestyle="-")
    plt.xscale("log")
    plt.xlabel("Number of hypersphere grid vectors")
    plt.ylabel("Validation loss")
    plt.title("Hypersphere grid projection validation loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_path, dpi=200)
    print(f"[demo] Saved validation loss plot to {args.plot_path}")

    summary = {
        "vector_counts": vector_counts,
        "losses": losses,
        "details": results,
        "eval_iters": args.eval_iters,
        "dataset": dataset_name,
        "grid_method": args.grid_method,
        "grid_mean": args.grid_mean,
        "grid_std": args.grid_std,
        "grid_normalize": args.grid_normalize,
    }

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[demo] Saved evaluation metrics to {args.metrics_path}")
if __name__ == "__main__":
    main()
