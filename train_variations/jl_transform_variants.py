# train_variations/jl_transform_variants.py
"""Utilities to apply Johnson-Lindenstrauss (JL) transforms during training.

This module mirrors the functionality of :mod:`initializations/jl_transform_ckpt.py`
but operates directly on an in-memory model.  It allows projecting all model
parameters to a new embedding dimension via several JL transform variants and
returns a freshly initialised :class:`~model.GPT` instance that can resume
training with the transformed weights.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch

from model import GPT
from gpt_conf import GPTConfig


def sign_matrix(out_dim: int, in_dim: int, generator: torch.Generator, device) -> torch.Tensor:
    """Create a sign-based JL projection matrix of shape (out_dim, in_dim)."""
    mat = torch.randint(0, 2, (out_dim, in_dim), generator=generator, device=device, dtype=torch.float32)
    mat = mat * 2 - 1
    mat /= math.sqrt(out_dim)
    return mat


def jl_project_tensor(tensor: torch.Tensor, proj: torch.Tensor, vertical_only: bool = False) -> torch.Tensor:
    """Project ``tensor`` using ``proj``.

    Parameters
    ----------
    tensor        : Tensor to project.
    proj          : Projection matrix of shape (out_dim, in_dim).
    vertical_only : If True, apply projection only along the first dimension.
    """
    in_dim = proj.shape[1]

    if tensor.ndim == 0:
        return tensor

    if vertical_only:
        if tensor.ndim > 1 and tensor.shape[0] == in_dim:
            return proj @ tensor
        if tensor.ndim == 1 and tensor.shape[0] == in_dim:
            return (proj @ tensor.unsqueeze(-1)).squeeze(-1)
        return tensor

    if tensor.ndim >= 1 and tensor.shape[-1] == in_dim:
        tensor = tensor @ proj.t()

    if tensor.ndim > 1 and tensor.shape[0] == in_dim:
        tensor = proj @ tensor
    elif tensor.ndim == 1 and tensor.shape[0] == in_dim:
        tensor = (proj @ tensor.unsqueeze(-1)).squeeze(-1)

    return tensor


def hadamard(n: int) -> torch.Tensor:
    """Return a Hadamard matrix of size ``n`` (``n`` must be a power of two)."""
    H = torch.tensor([[1.0]])
    size = 1
    while size < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        size *= 2
    return H


def build_projection(out_dim: int, in_dim: int, jl_type: str, g: torch.Generator, device) -> torch.Tensor:
    """Construct a JL projection matrix according to ``jl_type``."""
    if jl_type == "gaussian":
        proj = torch.empty((out_dim, in_dim), device=device)
        proj.normal_(mean=0.0, std=1.0, generator=g)
        proj /= math.sqrt(out_dim)
    elif jl_type == "sign":
        proj = sign_matrix(out_dim, in_dim, g, device)
    elif jl_type == "sparse":
        rand = torch.rand(out_dim, in_dim, generator=g, device=device)
        proj = torch.zeros_like(rand)
        proj[rand < 1 / 6] = 1
        proj[(rand >= 1 / 6) & (rand < 2 / 6)] = -1
        proj *= math.sqrt(3.0 / out_dim)
    elif jl_type == "srht":
        if (in_dim & (in_dim - 1)) != 0:
            raise ValueError("srht JL requires n_embd to be a power of two")
        D = torch.randint(0, 2, (in_dim,), generator=g, device=device, dtype=torch.float32) * 2 - 1
        H = hadamard(in_dim).to(device)
        idx = torch.randperm(in_dim, generator=g, device=device)[:out_dim]
        proj = H[idx] * D
        proj /= math.sqrt(out_dim)
    elif jl_type == "qr":
        raw = torch.randn(out_dim, in_dim, generator=g, device=device)
        q, _ = torch.linalg.qr(raw.T, mode="reduced")
        proj = q.T
        proj /= math.sqrt(out_dim)
    else:
        raise ValueError(f"Unknown jl_type: {jl_type}")
    return proj


def jl_transform_model(
    model: GPT,
    out_embd: int,
    jl_type: str = "gaussian",
    seed: int = 1337,
    cproj_vertical: bool = False,
) -> GPT:
    """Return a new ``GPT`` model whose parameters were JL-projected.

    The projection is applied even if ``out_embd`` equals the model's current
    embedding dimension.

    Parameters
    ----------
    model         : Existing GPT model to transform.
    out_embd      : Desired embedding dimension after projection.
    jl_type       : Variant of JL transform (gaussian, sign, sparse, srht, qr).
    seed          : Random seed for projection matrix.
    cproj_vertical: If True, project ``c_proj`` weights along the first dimension.
    """
    device = next(model.parameters()).device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    old_embd = model.config.n_embd
    proj = build_projection(out_embd, old_embd, jl_type, g, device)

    state_dict = model.state_dict()
    for key, tensor in list(state_dict.items()):
        if not torch.is_floating_point(tensor):
            continue
        vertical = cproj_vertical and key.endswith("c_proj.weight")
        state_dict[key] = jl_project_tensor(tensor, proj, vertical_only=vertical)

    cfg_dict = asdict(model.config)
    cfg_dict["n_embd"] = out_embd
    if cfg_dict.get("n_embd_wte") == old_embd:
        cfg_dict["n_embd_wte"] = out_embd

    new_conf = GPTConfig(**cfg_dict)
    new_model = GPT(new_conf)
    new_model.load_state_dict(state_dict, strict=False)
    return new_model
