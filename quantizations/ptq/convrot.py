"""Small, framework-independent ConvRot building blocks.

This module implements the *fake* W4A4 arithmetic needed by the ConvRot demo.
It intentionally keeps quantized values in floating point; it is an accuracy
demonstration, not an INT4 GEMM kernel.
"""

import math

import torch


def regular_hadamard(order: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return the normalized regular Hadamard matrix for ``order = 4**k``."""
    if order < 4:
        raise ValueError("regular Hadamard order must be at least 4")
    value = order
    while value > 1 and value % 4 == 0:
        value //= 4
    if value != 1:
        raise ValueError("regular Hadamard order must be a power of four")

    base = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        device=device,
        dtype=dtype or torch.float32,
    )
    matrix = base
    current_order = 4
    while current_order < order:
        matrix = torch.kron(matrix, base)
        current_order *= 4
    return matrix / math.sqrt(order)


def group_rotate(values: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply ``rotation`` independently to groups on the final dimension."""
    group_size = rotation.shape[0]
    if rotation.ndim != 2 or rotation.shape[1] != group_size:
        raise ValueError("rotation must be square")
    if values.shape[-1] % group_size:
        raise ValueError("last dimension must be divisible by the group size")
    groups = values.reshape(*values.shape[:-1], -1, group_size)
    return torch.matmul(groups, rotation).reshape_as(values)


def fake_symmetric_quantize(values: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per-vector symmetric fake quantization along the final dimension."""
    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = 2 ** (bits - 1) - 1
    scale = values.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.clamp(torch.round(values / scale), -qmax, qmax) * scale


def fake_w4a4_linear(
    activations: torch.Tensor,
    weight: torch.Tensor,
    *,
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a bias-free fake-W4A4 linear operation, optionally with ConvRot."""
    if rotation is not None:
        activations = group_rotate(activations, rotation)
        weight = group_rotate(weight, rotation)
    return fake_symmetric_quantize(activations) @ fake_symmetric_quantize(weight).T
