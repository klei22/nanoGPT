import argparse
import datetime
import os
import shutil
from typing import Dict, Any

import torch


L2_NORM_DIM = -1
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two nanoGPT checkpoints by L2-normalizing vectors, "
            "adding them, and L2-normalizing again."
        )
    )
    parser.add_argument(
        "ckpt_dir_a",
        type=str,
        help="Directory containing ckpt.pt from the first training run",
    )
    parser.add_argument(
        "ckpt_dir_b",
        type=str,
        help="Directory containing ckpt.pt from the second training run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write the merged checkpoint (defaults to <ckpt_dir_a>_merge)",
    )
    parser.add_argument(
        "--merge_mode",
        type=str,
        choices=("l2", "simple", "orthomerge"),
        default="l2",
        help=(
            "Merge mode to use. 'l2' matches the original behavior, "
            "'simple' averages tensors, and 'orthomerge' performs "
            "orthogonal-residual decoupling."
        ),
    )
    parser.add_argument(
        "--skip_final_norm_wte_lm_head",
        action="store_true",
        help="Skip the final L2 normalization for wte/lm_head weights",
    )
    parser.add_argument(
        "--no_l2_normalize",
        action="store_true",
        help=(
            "Disable all L2 normalizations and instead add and divide by --simple_divisor."
        ),
    )
    parser.add_argument(
        "--simple_divisor",
        type=float,
        default=2.0,
        help=(
            "Divisor for simple merging (used only when --no_l2_normalize is set)."
        ),
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help=(
            "Directory containing the base ckpt.pt for orthomerge mode. "
            "Required when --merge_mode=orthomerge."
        ),
    )
    parser.add_argument(
        "--orthomerge_max_dim",
        type=int,
        default=2048,
        help=(
            "Maximum output dimension to apply orthogonal merging. Larger "
            "matrices fall back to task-vector merging."
        ),
    )
    return parser.parse_args()


def l2_normalize(tensor: torch.Tensor, dim: int = L2_NORM_DIM) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1:
        dim = 0
    norm = tensor.norm(dim=dim, keepdim=True).clamp_min(EPS)
    return tensor / norm


def is_wte_or_lm_head(key: str) -> bool:
    parts = key.split(".")
    for part in parts:
        if part == "wte" or part.startswith("wte_"):
            return True
        if part == "lm_head" or part.startswith("lm_head_"):
            return True
    return False


def load_checkpoint(ckpt_dir: str) -> Dict[str, Any]:
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def merge_l2(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    skip_final_norm: bool = False,
) -> torch.Tensor:
    norm_a = l2_normalize(tensor_a)
    norm_b = l2_normalize(tensor_b)
    merged = norm_a + norm_b
    if skip_final_norm:
        return merged
    return l2_normalize(merged)


def merge_simple(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    return (tensor_a + tensor_b) / divisor


def orthogonal_procrustes(
    target: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    cross = target @ base.T
    u, _, v_h = torch.linalg.svd(cross, full_matrices=False)
    return u @ v_h


def inverse_cayley(r_mat: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(r_mat.shape[0], device=r_mat.device, dtype=r_mat.dtype)
    numerator = r_mat - identity
    denominator = r_mat + identity
    return torch.linalg.solve(denominator.T, numerator.T).T


def cayley(q_mat: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(q_mat.shape[0], device=q_mat.device, dtype=q_mat.dtype)
    numerator = identity + q_mat
    denominator = identity - q_mat
    return torch.linalg.solve(denominator.T, numerator.T).T


def should_use_orthomerge(
    key: str,
    tensor: torch.Tensor,
    max_dim: int,
) -> bool:
    if tensor.ndim != 2:
        return False
    if is_wte_or_lm_head(key):
        return False
    return tensor.shape[0] <= max_dim


def main() -> None:
    args = parse_args()

    checkpoint_a = load_checkpoint(args.ckpt_dir_a)
    checkpoint_b = load_checkpoint(args.ckpt_dir_b)
    checkpoint_base = None
    if args.merge_mode == "orthomerge":
        if not args.base_ckpt_dir:
            raise ValueError("--base_ckpt_dir is required for --merge_mode=orthomerge")
        checkpoint_base = load_checkpoint(args.base_ckpt_dir)

    state_dict_a = checkpoint_a.get("model", checkpoint_a)
    state_dict_b = checkpoint_b.get("model", checkpoint_b)
    state_dict_base = (
        checkpoint_base.get("model", checkpoint_base)
        if checkpoint_base is not None
        else None
    )

    if state_dict_a.keys() != state_dict_b.keys():
        missing_a = sorted(set(state_dict_b.keys()) - set(state_dict_a.keys()))
        missing_b = sorted(set(state_dict_a.keys()) - set(state_dict_b.keys()))
        raise ValueError(
            "Checkpoint parameter keys do not match. "
            f"Missing in A: {missing_a[:5]}{'...' if len(missing_a) > 5 else ''}. "
            f"Missing in B: {missing_b[:5]}{'...' if len(missing_b) > 5 else ''}."
        )
    if state_dict_base is not None and state_dict_a.keys() != state_dict_base.keys():
        missing_base = sorted(set(state_dict_a.keys()) - set(state_dict_base.keys()))
        raise ValueError(
            "Base checkpoint parameter keys do not match. "
            f"Missing in base: {missing_base[:5]}{'...' if len(missing_base) > 5 else ''}."
        )

    merged_state_dict = {}
    for key, tensor_a in state_dict_a.items():
        tensor_b = state_dict_b[key]
        tensor_base = state_dict_base[key] if state_dict_base is not None else None
        if not torch.is_floating_point(tensor_a):
            if tensor_a.shape != tensor_b.shape or not torch.equal(tensor_a, tensor_b):
                raise ValueError(f"Non-floating tensor mismatch for key {key}")
            merged_state_dict[key] = tensor_a
            continue

        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tensor_a.shape} vs {tensor_b.shape}"
            )

        merge_mode = args.merge_mode
        if args.no_l2_normalize:
            merge_mode = "simple"

        if merge_mode == "orthomerge":
            if tensor_base is None:
                raise ValueError("Base checkpoint is required for orthomerge.")

            if should_use_orthomerge(key, tensor_a, args.orthomerge_max_dim):
                r_a = orthogonal_procrustes(tensor_a, tensor_base)
                r_b = orthogonal_procrustes(tensor_b, tensor_base)
                q_a = inverse_cayley(r_a)
                q_b = inverse_cayley(r_b)
                q_sum = q_a + q_b
                q_sum_norm = q_sum.norm().clamp_min(EPS)
                q_scale = (q_a.norm() + q_b.norm()) / q_sum_norm
                q_merged = q_scale * (q_sum / 2.0)
                r_merged = cayley(q_merged)
                ortho_component = r_merged @ tensor_base
                residual_a = tensor_a - r_a @ tensor_base
                residual_b = tensor_b - r_b @ tensor_base
                residual_merged = merge_l2(
                    residual_a,
                    residual_b,
                    skip_final_norm=(
                        args.skip_final_norm_wte_lm_head and is_wte_or_lm_head(key)
                    ),
                )
                merged_state_dict[key] = ortho_component + residual_merged
            else:
                task_a = tensor_a - tensor_base
                task_b = tensor_b - tensor_base
                merged_task = merge_l2(
                    task_a,
                    task_b,
                    skip_final_norm=(
                        args.skip_final_norm_wte_lm_head and is_wte_or_lm_head(key)
                    ),
                )
                merged_state_dict[key] = tensor_base + merged_task
            continue

        if merge_mode == "simple":
            merged_state_dict[key] = merge_simple(
                tensor_a,
                tensor_b,
                args.simple_divisor,
            )
            continue

        merged_state_dict[key] = merge_l2(
            tensor_a,
            tensor_b,
            skip_final_norm=(
                args.skip_final_norm_wte_lm_head and is_wte_or_lm_head(key)
            ),
        )

    if isinstance(checkpoint_a, dict) and "model" in checkpoint_a:
        checkpoint_a["model"] = merged_state_dict
    else:
        checkpoint_a = merged_state_dict

    checkpoint_a.pop("optimizer", None)
    checkpoint_a.pop("scheduler", None)
    if isinstance(checkpoint_a, dict):
        checkpoint_a["iter_num"] = 0
        checkpoint_a["best_val_loss"] = 1e9
        checkpoint_a["best_iter"] = 0
        checkpoint_a["best_tokens"] = 0

    out_dir = args.out_dir or f"{args.ckpt_dir_a.rstrip('/').rstrip(os.sep)}_merge"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint_a, os.path.join(out_dir, "ckpt.pt"))

    meta_path = os.path.join(args.ckpt_dir_a, "meta.pkl")
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(out_dir, "meta.pkl"))

    print(
        "✔ Merged checkpoint written to "
        f"{out_dir} at {datetime.datetime.now().isoformat(timespec='seconds')}"
    )


if __name__ == "__main__":
    main()
