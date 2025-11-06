# train_variations/distillation_loss_variants.py
"""Knowledge distillation loss variants used during training."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


def _reshape_and_mask(values: torch.Tensor, targets: Optional[torch.Tensor]) -> torch.Tensor:
    """Return a flat tensor of values with ignore_index positions removed."""

    flat = values.reshape(-1)
    if targets is None:
        return flat

    mask = (targets != -1).reshape(-1)
    if mask.any():
        return flat[mask]
    return flat.new_empty(0)


def _masked_mean(values: torch.Tensor, targets: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Compute the mean over valid positions, returning ``None`` if empty."""

    masked = _reshape_and_mask(values, targets)
    if masked.numel() == 0:
        return None
    return masked.mean()


def _forward_kl_tokenwise(
    student_scaled: torch.Tensor,
    teacher_scaled: torch.Tensor,
) -> torch.Tensor:
    """Token-wise KL(teacher || student)."""

    student_log_probs = F.log_softmax(student_scaled, dim=-1)
    teacher_probs = F.softmax(teacher_scaled, dim=-1)
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    return kl_per_token.sum(dim=-1)


def _reverse_kl_tokenwise(
    student_scaled: torch.Tensor,
    teacher_scaled: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Token-wise KL(student || teacher)."""

    student_probs = F.softmax(student_scaled, dim=-1)
    teacher_probs = F.softmax(teacher_scaled, dim=-1)

    log_student = torch.log(student_probs + eps)
    log_teacher = torch.log(teacher_probs + eps)
    return torch.sum(student_probs * (log_student - log_teacher), dim=-1)


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Standard forward KL distillation loss (teacher || student)."""

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature
    token_kl = _forward_kl_tokenwise(student_scaled, teacher_scaled)
    mean = _masked_mean(token_kl, targets)
    if mean is None:
        return student_logits.new_zeros(())
    loss = mean * (temperature ** 2)
    return loss.to(student_logits.dtype)


def reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reverse KL distillation (student || teacher)."""

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature
    token_kl = _reverse_kl_tokenwise(student_scaled, teacher_scaled, eps)
    mean = _masked_mean(token_kl, targets)
    if mean is None:
        return student_logits.new_zeros(())
    loss = mean * (temperature ** 2)
    return loss.to(student_logits.dtype)


def symmetric_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Symmetric KL: ½( KL(t‖s) + KL(s‖t) )."""

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature
    forward = _forward_kl_tokenwise(student_scaled, teacher_scaled)
    reverse = _reverse_kl_tokenwise(student_scaled, teacher_scaled, eps)
    combined = 0.5 * (forward + reverse)
    mean = _masked_mean(combined, targets)
    if mean is None:
        return student_logits.new_zeros(())
    loss = mean * (temperature ** 2)
    return loss.to(student_logits.dtype)


def jensen_shannon_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Jensen-Shannon divergence between teacher and student distributions."""

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature

    student_probs = F.softmax(student_scaled, dim=-1)
    teacher_probs = F.softmax(teacher_scaled, dim=-1)
    mixture = 0.5 * (student_probs + teacher_probs)

    log_student = torch.log(student_probs + eps)
    log_teacher = torch.log(teacher_probs + eps)
    log_mixture = torch.log(mixture + eps)

    teacher_term = torch.sum(teacher_probs * (log_teacher - log_mixture), dim=-1)
    student_term = torch.sum(student_probs * (log_student - log_mixture), dim=-1)
    token_js = 0.5 * (teacher_term + student_term)

    mean = _masked_mean(token_js, targets)
    if mean is None:
        return student_logits.new_zeros(())
    loss = mean * (temperature ** 2)
    return loss.to(student_logits.dtype)


def logit_mse_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean-squared-error between temperature-scaled logits."""

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature
    token_mse = F.mse_loss(student_scaled, teacher_scaled, reduction="none").mean(dim=-1)
    mean = _masked_mean(token_mse, targets)
    if mean is None:
        return student_logits.new_zeros(())
    return mean.to(student_logits.dtype)


DISTILLATION_LOSS_VARIANTS: Dict[str, Callable[..., torch.Tensor]] = {
    "kl_divergence": kl_divergence_loss,
    "reverse_kl": reverse_kl_loss,
    "symmetric_kl": symmetric_kl_loss,
    "jensen_shannon": jensen_shannon_loss,
    "logit_mse": logit_mse_loss,
}


def build_distillation_loss(args) -> Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Construct the distillation loss callable based on CLI arguments."""

    variant_name = getattr(args, "distillation_loss", None)
    if not variant_name:
        return None

    if variant_name not in DISTILLATION_LOSS_VARIANTS:
        raise ValueError(
            f"Unknown distillation loss '{variant_name}'. Available: {list(DISTILLATION_LOSS_VARIANTS.keys())}"
        )

    temperature = float(getattr(args, "distillation_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("distillation_temperature must be positive")

    eps = float(getattr(args, "distillation_eps", 1e-8))
    if eps <= 0:
        raise ValueError("distillation_eps must be positive")

    base_loss = DISTILLATION_LOSS_VARIANTS[variant_name]

    def loss_fn(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, *, iter_num=None):
        return base_loss(
            student_logits,
            teacher_logits,
            targets,
            iter_num=iter_num,
            temperature=temperature,
            eps=eps,
        )

    return loss_fn


__all__ = [
    "DISTILLATION_LOSS_VARIANTS",
    "build_distillation_loss",
]

