"""Demo script to evaluate hypersphere grid snapping at different scales."""

import argparse
import json
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import torch

from model import GPT, GPTConfig
from sample import (
    apply_hypersphere_grid_snap,
    calculate_validation_loss,
    generate_hypersphere_grid,
    load_validation_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hypersphere grid snapping over multiple grid sizes.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint produced during training (expects ckpt.pt format)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="hypersphere_grid_demo",
        help="Directory where plots and metrics will be written",
    )
    parser.add_argument(
        "--grid_sizes",
        type=int,
        nargs="+",
        default=[10_000, 100_000, 1_000_000],
        help="List of grid sizes to evaluate",
    )
    parser.add_argument(
        "--grid_method",
        type=str,
        default="random",
        choices=["random"],
        help="Method used to construct the hypersphere grid",
    )
    parser.add_argument(
        "--grid_mean",
        type=float,
        default=0.0,
        help="Mean of the Gaussian used to generate random grid vectors",
    )
    parser.add_argument(
        "--grid_std",
        type=float,
        default=0.02,
        help="Standard deviation of the Gaussian used to generate random grid vectors",
    )
    parser.add_argument(
        "--grid_seed",
        type=int,
        default=0,
        help="Base seed used when generating grids",
    )
    parser.add_argument(
        "--grid_seed_step",
        type=int,
        default=1,
        help="Amount added to the grid seed for each successive grid size",
    )
    parser.add_argument(
        "--grid_targets",
        type=str,
        nargs="+",
        default=["block", "attn", "mlp"],
        choices=["block", "attn", "mlp"],
        help="Pre-normalization modules that should snap to the grid",
    )
    parser.add_argument(
        "--grid_match_norm",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Preserve the norm of the pre-normalized activations when snapping",
    )
    parser.add_argument(
        "--grid_normalize",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Normalize grid vectors to unit length before snapping",
    )
    parser.add_argument(
        "--grid_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Override dtype used for hypersphere grid tensors",
    )
    parser.add_argument(
        "--grid_chunk_size",
        type=int,
        default=65_536,
        help="Chunk size for overlap computation with the hypersphere grid",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Dataset name used to locate data/<dataset>/val.bin",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=50,
        help="Number of validation iterations per grid size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for evaluation (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Autocast dtype used during evaluation",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Optional explicit path for the validation loss plot",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for PyTorch random number generation",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _resolve_grid_dtype(name: str | None, fallback: torch.dtype) -> torch.dtype:
    if name is None:
        return fallback
    return _resolve_dtype(name)


def _load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist")
    return torch.load(ckpt_path, map_location=device)


def _build_model(checkpoint: Dict[str, torch.Tensor]) -> GPT:
    if "model_args" not in checkpoint or "model" not in checkpoint:
        raise KeyError("Checkpoint is missing required 'model_args' or 'model' entries")
    model_args = dict(checkpoint["model_args"])
    model_args["dropout"] = 0.0
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate_grid_size(
    *,
    ckpt_path: str,
    device: torch.device,
    autocast_dtype: torch.dtype,
    grid_size: int,
    grid_method: str,
    grid_mean: float,
    grid_std: float,
    grid_seed: int,
    grid_dtype: torch.dtype,
    grid_targets: Sequence[str],
    grid_match_norm: bool,
    grid_normalize: bool,
    grid_chunk_size: int,
    eval_dataset: str,
    eval_iters: int,
) -> Dict[str, float]:
    checkpoint = _load_checkpoint(ckpt_path, device)
    model = _build_model(checkpoint)
    model.to(device)
    model.eval()

    grid = generate_hypersphere_grid(
        grid_size,
        model.config.n_embd,
        method=grid_method,
        mean=grid_mean,
        std=grid_std,
        seed=grid_seed,
        device=device,
        dtype=grid_dtype,
    )

    apply_hypersphere_grid_snap(
        model,
        grid,
        targets=grid_targets,
        match_input_norm=grid_match_norm,
        normalize_grid=grid_normalize,
        chunk_size=grid_chunk_size,
    )

    val_data = load_validation_data(model.config.block_size, eval_dataset)
    metrics = calculate_validation_loss(
        model,
        val_data,
        model.config.block_size,
        eval_iters,
        device.type,
        autocast_dtype,
    )

    return metrics


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    autocast_dtype = _resolve_dtype(args.dtype)
    grid_dtype = _resolve_grid_dtype(args.grid_dtype, autocast_dtype)

    all_metrics: List[Dict[str, float]] = []
    val_losses: List[float] = []

    for idx, size in enumerate(args.grid_sizes):
        seed = args.grid_seed + idx * args.grid_seed_step
        print(f"Evaluating hypersphere grid snapping with {size} vectors (seed={seed})...")
        metrics = evaluate_grid_size(
            ckpt_path=args.ckpt,
            device=device,
            autocast_dtype=autocast_dtype,
            grid_size=size,
            grid_method=args.grid_method,
            grid_mean=args.grid_mean,
            grid_std=args.grid_std,
            grid_seed=seed,
            grid_dtype=grid_dtype,
            grid_targets=args.grid_targets,
            grid_match_norm=args.grid_match_norm,
            grid_normalize=args.grid_normalize,
            grid_chunk_size=args.grid_chunk_size,
            eval_dataset=args.eval_dataset,
            eval_iters=args.eval_iters,
        )
        metrics["grid_size"] = size
        metrics["grid_seed"] = seed
        all_metrics.append(metrics)
        val_losses.append(metrics.get("val", float("nan")))

        # Free CUDA memory between evaluations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_path = args.plot_path or os.path.join(args.out_dir, "hypersphere_grid_val_loss.png")
    plt.figure(figsize=(8, 5))
    plt.plot(args.grid_sizes, val_losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Grid size (log scale)")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs. hypersphere grid size")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved validation loss plot to {plot_path}")

    metrics_path = args.metrics_path or os.path.join(args.out_dir, "hypersphere_grid_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"Saved metrics to {metrics_path}")
if __name__ == "__main__":
    main()
