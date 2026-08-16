"""Optimizer-independent component-gradient diagnostics.

The caller supplies one scalar loss per fixed component microbatch.  Gradients
are obtained with ``autograd.grad`` so existing ``parameter.grad`` buffers and
optimizer state are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


DEFAULT_PARAMETER_PATTERNS = ("transformer.h.", "transformer.ln_f", "scale_down", "lm_head")


def select_diagnostic_parameters(
    model: torch.nn.Module,
    patterns: Sequence[str] = DEFAULT_PARAMETER_PATTERNS,
) -> tuple[torch.nn.Parameter, ...]:
    """Select the final block/norm/adapter/head subset without duplicates."""

    named = list(model.named_parameters())
    block_indices = [
        int(name.split("transformer.h.", 1)[1].split(".", 1)[0])
        for name, _ in named
        if "transformer.h." in name
    ]
    last_block = f"transformer.h.{max(block_indices)}." if block_indices else None
    selected = []
    seen = set()
    for name, parameter in named:
        matches = any(pattern in name for pattern in patterns[1:])
        matches = matches or (last_block is not None and last_block in name)
        if matches and parameter.requires_grad and id(parameter) not in seen:
            selected.append(parameter)
            seen.add(id(parameter))
    return tuple(selected)


@dataclass(frozen=True)
class GradientNoiseDiagnostics:
    mean_squared_norm: float
    squared_mean_norm: float
    noise_variance: float
    coherence: float
    noise_scale: float
    microbatch_count: int


def component_gradient_diagnostics(
    losses: Iterable[torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    eps: float = 1e-12,
) -> GradientNoiseDiagnostics:
    """Measure component-gradient agreement without modifying model gradients.

    ``losses`` must come from equal-sized microbatches. Parameters unused by a
    component are treated as having a zero gradient for that component.
    """

    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not params:
        raise ValueError("at least one trainable diagnostic parameter is required")

    sum_gradients = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in params]
    squared_norm_sum = torch.zeros((), device=params[0].device, dtype=torch.float64)
    count = 0

    for loss in losses:
        gradients = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        component_squared_norm = torch.zeros_like(squared_norm_sum)
        for total, gradient in zip(sum_gradients, gradients):
            if gradient is None:
                continue
            gradient_fp32 = gradient.detach().float()
            total.add_(gradient_fp32)
            component_squared_norm += gradient_fp32.double().square().sum()
        squared_norm_sum += component_squared_norm
        count += 1

    if count < 2:
        raise ValueError("at least two component microbatch losses are required")

    mean_squared_norm = squared_norm_sum / count
    squared_mean_norm = sum(
        (gradient_sum.double() / count).square().sum() for gradient_sum in sum_gradients
    )
    noise_variance = count / (count - 1) * torch.clamp(
        mean_squared_norm - squared_mean_norm, min=0.0
    )
    coherence = squared_mean_norm / (mean_squared_norm + eps)
    noise_scale = mean_squared_norm / (squared_mean_norm + eps)

    return GradientNoiseDiagnostics(
        mean_squared_norm=float(mean_squared_norm.item()),
        squared_mean_norm=float(squared_mean_norm.item()),
        noise_variance=float(noise_variance.item()),
        coherence=float(coherence.item()),
        noise_scale=float(noise_scale.item()),
        microbatch_count=count,
    )
