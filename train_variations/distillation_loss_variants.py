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


def mixed_hard_soft_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temperature: float = 4.0,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Classic Hinton distillation: alpha * CE(hard) + (1-alpha) * T^2 * KL(teacher||student).

    ``alpha`` controls the balance between the hard-label cross-entropy and the
    soft-label KL term.  ``temperature`` softens both distributions; the KL
    term is scaled by ``T^2`` to preserve gradient magnitudes as in the
    original paper (Hinton et al. 2015).
    """

    # Hard-label CE (no temperature)
    if targets is not None:
        hard_ce = F.cross_entropy(
            student_logits.float().view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
    else:
        hard_ce = student_logits.new_zeros(())

    # Soft KL at elevated temperature
    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature
    token_kl = _forward_kl_tokenwise(student_scaled, teacher_scaled)
    mean_kl = _masked_mean(token_kl, targets)
    if mean_kl is None:
        soft_loss = student_logits.new_zeros(())
    else:
        soft_loss = mean_kl * (temperature ** 2)

    loss = alpha * hard_ce + (1.0 - alpha) * soft_loss
    return loss.to(student_logits.dtype)


def top_k_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temperature: float = 1.0,
    top_k: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sparse forward KL that only distills from the teacher's top-k tokens.

    Restricting distillation to the teacher's most probable tokens filters out
    noisy low-probability mass and reduces memory overhead for large
    vocabularies.
    """

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature

    # Build a sparse teacher distribution over the top-k teacher tokens
    with torch.no_grad():
        teacher_probs_full = F.softmax(teacher_scaled, dim=-1)
        topk_vals, topk_idx = teacher_probs_full.topk(top_k, dim=-1)
        # Re-normalise the truncated teacher distribution
        topk_sum = topk_vals.sum(dim=-1, keepdim=True).clamp(min=eps)
        topk_probs = topk_vals / topk_sum  # (..., top_k)

    # Student log-probs at the same top-k positions
    student_log_probs = F.log_softmax(student_scaled, dim=-1)
    student_log_topk = student_log_probs.gather(-1, topk_idx)  # (..., top_k)

    # KL(truncated teacher || student at top-k positions)
    token_kl = (topk_probs * (torch.log(topk_probs + eps) - student_log_topk)).sum(dim=-1)

    mean = _masked_mean(token_kl, targets)
    if mean is None:
        return student_logits.new_zeros(())
    loss = mean * (temperature ** 2)
    return loss.to(student_logits.dtype)


def adaptive_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    *,
    iter_num: int | None = None,
    temp_start: float = 4.0,
    temp_end: float = 1.0,
    temp_steps: int = 10000,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Forward KL with a temperature that linearly anneals from ``temp_start``
    down to ``temp_end`` over ``temp_steps`` training iterations.

    High temperature early in training encourages the student to match the
    teacher's broad distribution; as temperature decreases the student is
    pushed toward sharper, more confident agreement.
    """

    if iter_num is not None and temp_steps > 0:
        progress = min(float(iter_num) / float(temp_steps), 1.0)
        temperature = temp_start + (temp_end - temp_start) * progress
    else:
        temperature = temp_start

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature
    token_kl = _forward_kl_tokenwise(student_scaled, teacher_scaled)
    mean = _masked_mean(token_kl, targets)
    if mean is None:
        return student_logits.new_zeros(())
    loss = mean * (temperature ** 2)
    return loss.to(student_logits.dtype)


DISTILLATION_LOSS_VARIANTS: Dict[str, Callable[..., torch.Tensor]] = {
    "kl_divergence": kl_divergence_loss,
    "reverse_kl": reverse_kl_loss,
    "symmetric_kl": symmetric_kl_loss,
    "jensen_shannon": jensen_shannon_loss,
    "logit_mse": logit_mse_loss,
    "mixed_hard_soft": mixed_hard_soft_loss,
    "top_k_kl": top_k_kl_loss,
    "adaptive_kl": adaptive_kl_loss,
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

    if variant_name == "mixed_hard_soft":
        alpha = float(getattr(args, "distillation_alpha", 0.5))

        def loss_fn(student_logits, teacher_logits, targets, *, iter_num=None):
            return base_loss(
                student_logits, teacher_logits, targets,
                iter_num=iter_num, temperature=temperature, alpha=alpha, eps=eps,
            )

    elif variant_name == "top_k_kl":
        top_k = int(getattr(args, "distillation_top_k", 50))

        def loss_fn(student_logits, teacher_logits, targets, *, iter_num=None):
            return base_loss(
                student_logits, teacher_logits, targets,
                iter_num=iter_num, temperature=temperature, top_k=top_k, eps=eps,
            )

    elif variant_name == "adaptive_kl":
        temp_start = float(getattr(args, "distillation_temp_start", 4.0))
        temp_end = float(getattr(args, "distillation_temp_end", 1.0))
        temp_steps = int(getattr(args, "distillation_temp_steps", 10000))

        def loss_fn(student_logits, teacher_logits, targets, *, iter_num=None):
            return base_loss(
                student_logits, teacher_logits, targets,
                iter_num=iter_num, temp_start=temp_start, temp_end=temp_end,
                temp_steps=temp_steps, eps=eps,
            )

    else:
        def loss_fn(student_logits, teacher_logits, targets, *, iter_num=None):
            return base_loss(
                student_logits, teacher_logits, targets,
                iter_num=iter_num, temperature=temperature, eps=eps,
            )

    return loss_fn


__all__ = [
    "DISTILLATION_LOSS_VARIANTS",
    "build_distillation_loss",
]

