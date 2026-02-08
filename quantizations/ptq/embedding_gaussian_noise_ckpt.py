import argparse
import os
import shutil
from typing import Iterable, List, Optional

import torch


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply embedding-style Gaussian vector noise to all weights in a checkpoint "
            "(vector mode only)."
        )
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and meta.pkl from a previous training run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Directory to write the noisy checkpoint(s) (defaults to "
            "<ckpt_dir>_gaussian_noise). If multiple alphas are provided, "
            "subdirectories are created under this path."
        ),
    )
    parser.add_argument(
        "--alphas",
        type=str,
        nargs="+",
        default=["0.2"],
        help=(
            "Comma- or space-separated alpha values to scale the noise (default: 0.2). "
            "Example: --alphas 0.1 0.2 or --alphas 0.1,0.2"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for the Gaussian noise",
    )
    return parser.parse_args()


def parse_alpha_list(items: Iterable[str]) -> List[float]:
    alphas: List[float] = []
    for item in items:
        for part in str(item).split(","):
            value = part.strip()
            if not value:
                continue
            alphas.append(float(value))
    if not alphas:
        raise ValueError("At least one alpha value must be provided")
    return alphas


def iter_state_items(state_dict):
    if isinstance(state_dict, torch.nn.Module):
        iterable = state_dict.state_dict().items()
    elif isinstance(state_dict, dict):
        iterable = state_dict.items()
    else:
        iterable = getattr(state_dict, "state_dict", lambda: {})().items()

    for key, value in iterable:
        if torch.is_tensor(value):
            yield key, value


def infer_embedding_dimension(checkpoint, state_dict) -> Optional[int]:
    for container_name in ("model_args", "config"):
        container = getattr(checkpoint, "get", None)
        if callable(container):
            container = checkpoint.get(container_name)
        else:
            container = None
        if isinstance(container, dict):
            value = container.get("n_embd")
            if isinstance(value, int):
                return value

    state_get = getattr(state_dict, "get", None)
    for search_key in (
        "transformer.wte.weight",
        "wte.weight",
        "tok_embeddings.weight",
    ):
        tensor = state_get(search_key) if callable(state_get) else None
        if torch.is_tensor(tensor) and tensor.ndim == 2:
            return int(tensor.shape[1])

    for name, tensor in iter_state_items(state_dict):
        if name.endswith("wte.weight") and torch.is_tensor(tensor) and tensor.ndim == 2:
            return int(tensor.shape[1])

    return None


def _vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)


def apply_noise_to_vectors(
    vectors: torch.Tensor,
    alphas: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    alphas = alphas.to(dtype=vectors.dtype)
    noise = torch.randn(
        vectors.shape,
        generator=generator,
        device=vectors.device,
        dtype=vectors.dtype,
    )
    noise = noise / (_vector_norm(noise) + EPS)
    weight_norm = _vector_norm(vectors)
    scaled_noise = noise.unsqueeze(0) * alphas.view(-1, *([1] * vectors.ndim))
    scaled_noise = scaled_noise * weight_norm.unsqueeze(0)
    perturbed = vectors.unsqueeze(0) + scaled_noise
    perturbed_norm = _vector_norm(perturbed)
    perturbed = perturbed / (perturbed_norm + EPS) * weight_norm.unsqueeze(0)
    return perturbed


def apply_noise_per_vector(
    tensor: torch.Tensor,
    alphas: torch.Tensor,
    embedding_dim: int,
    *,
    generator: torch.Generator,
) -> Optional[List[torch.Tensor]]:
    if tensor.ndim >= 1 and tensor.shape[-1] == embedding_dim:
        perturbed = apply_noise_to_vectors(tensor, alphas, generator=generator)
        return [perturbed[idx] for idx in range(perturbed.shape[0])]

    if tensor.ndim > 1 and tensor.shape[0] == embedding_dim:
        moved = torch.movedim(tensor, 0, -1)
        perturbed = apply_noise_to_vectors(moved, alphas, generator=generator)
        return [torch.movedim(perturbed[idx], -1, 0) for idx in range(perturbed.shape[0])]

    return None


def build_noisy_state_dicts(
    state_dict,
    alphas: List[float],
    embedding_dim: int,
    *,
    generator: torch.Generator,
) -> List[dict]:
    alpha_tensor = torch.tensor(alphas, dtype=torch.float32)
    noisy_state_dicts = [dict() for _ in alphas]
    for key, value in state_dict.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            for idx in range(len(alphas)):
                noisy_state_dicts[idx][key] = value
            continue
        outputs = apply_noise_per_vector(
            value, alpha_tensor, embedding_dim, generator=generator
        )
        if outputs is None:
            for idx in range(len(alphas)):
                noisy_state_dicts[idx][key] = value
            continue
        for idx, noisy in enumerate(outputs):
            noisy_state_dicts[idx][key] = noisy
    return noisy_state_dicts


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    alphas = parse_alpha_list(args.alphas)

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_obj = checkpoint["model"]
    else:
        state_obj = checkpoint

    if isinstance(state_obj, dict):
        state_dict = state_obj
    else:
        to_state_dict = getattr(state_obj, "state_dict", None)
        if callable(to_state_dict):
            state_dict = to_state_dict()
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint["model"] = state_dict
            else:
                checkpoint = state_dict
        else:
            raise TypeError(
                "Unsupported checkpoint format: expected a mapping for the model state"
            )

    embedding_dim = infer_embedding_dimension(checkpoint, state_dict)
    if embedding_dim is None:
        raise ValueError("Could not determine n_embd from checkpoint")

    g = torch.Generator()
    g.manual_seed(args.seed)

    noisy_state_dicts = build_noisy_state_dicts(
        state_dict, alphas, embedding_dim, generator=g
    )

    base_out_dir = args.out_dir or f"{args.ckpt_dir}_gaussian_noise"
    for alpha, noisy_state in zip(alphas, noisy_state_dicts):
        if len(alphas) == 1:
            out_dir = base_out_dir
        else:
            out_dir = os.path.join(base_out_dir, f"alpha_{format_alpha(alpha)}")
        os.makedirs(out_dir, exist_ok=True)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint["model"] = noisy_state
            out_checkpoint = checkpoint
        else:
            out_checkpoint = noisy_state

        torch.save(out_checkpoint, os.path.join(out_dir, "ckpt.pt"))

        meta_in = os.path.join(args.ckpt_dir, "meta.pkl")
        meta_out = os.path.join(out_dir, "meta.pkl")
        if os.path.exists(meta_in):
            shutil.copy(meta_in, meta_out)


if __name__ == "__main__":
    main()
