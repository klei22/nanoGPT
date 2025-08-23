# train_variations/loss_variants.py
"""Collection of loss functions and scheduling utilities.

This module provides a dictionary mapping loss names to callables. Each
loss takes ``logits`` and ``targets`` tensors and returns a scalar loss.
Optionally the current training iteration ``iter_num`` can be supplied
for schedulers or adaptive losses.

The default loss is standard cross entropy, but additional options are
provided that more strongly encourage correct top-1 predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss implementations
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, *, iter_num: int | None = None) -> torch.Tensor:
    """Standard cross-entropy loss used by the original codebase."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def label_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Cross entropy with label smoothing to prevent overconfidence."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
        label_smoothing=smoothing,
    )


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss from classification literature to focus on hard examples."""
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    with torch.no_grad():
        pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss[targets_flat != -1].mean()


def top1_focus_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Cross entropy with an extra penalty for wrong top-1 predictions."""
    ce = cross_entropy_loss(logits, targets)
    top1 = torch.argmax(logits, dim=-1)
    correct_top1 = (top1 == targets).float()
    penalty = 1.0 - correct_top1
    return ce + alpha * penalty.mean()


def top1_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    margin: float = 0.1,
) -> torch.Tensor:
    """Encourage the target logit to exceed others by a margin."""
    ce = cross_entropy_loss(logits, targets)
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    target_logits = logits_flat[torch.arange(logits_flat.size(0)), targets_flat]
    others = logits_flat.clone()
    others[torch.arange(logits_flat.size(0)), targets_flat] = float("-inf")
    max_other, _ = others.max(dim=-1)
    margin_loss = torch.clamp(margin - (target_logits - max_other), min=0.0)
    return ce + margin_loss.mean()


def entropy_penalty_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    beta: float = 0.01,
) -> torch.Tensor:
    """Cross entropy plus penalty on prediction entropy to encourage peaky outputs."""
    ce = cross_entropy_loss(logits, targets)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    mask = targets != -1
    return ce + beta * entropy[mask].mean()


def top1_ratio_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    beta: float = 0.5,
) -> torch.Tensor:
    """Novel loss encouraging the target logit to dominate all others exponentially."""
    ce = cross_entropy_loss(logits, targets)
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    mask = targets_flat != -1
    logits_flat = logits_flat[mask]
    targets_flat = targets_flat[mask]
    target_logits = logits_flat[torch.arange(logits_flat.size(0)), targets_flat]
    others = logits_flat.clone()
    others[torch.arange(logits_flat.size(0)), targets_flat] = float("-inf")
    max_other, _ = others.max(dim=-1)
    ratio_penalty = torch.exp(max_other - target_logits)
    return ce + beta * ratio_penalty.mean()


def rank_distance_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Scale loss by how far the target's rank is from top-1."""
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1
    with torch.no_grad():
        logits_sel = logits_flat[mask]
        targets_sel = targets_flat[mask]
        target_logits = logits_sel[torch.arange(logits_sel.size(0)), targets_sel]
        rank = (logits_sel > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
        scale = 1 + gamma * (rank.float() - 1)
        scale = torch.nan_to_num(scale, nan=0.0, posinf=1e4, neginf=0.0)
        scale = scale.clamp(max=1e4)
    scaled = torch.zeros_like(loss)
    scaled[mask] = loss[mask] * scale
    return scaled[mask].mean()


def flatness_boost_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Boost loss when the predicted distribution is flat (high entropy)."""
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1
    with torch.no_grad():
        probs = torch.softmax(logits_flat[mask], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        entropy_norm = entropy / math.log(v)
        scale = 1 + beta * entropy_norm
    scaled = torch.zeros_like(loss)
    scaled[mask] = loss[mask] * scale
    return scaled[mask].mean()


def entropy_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 2.0,
    beta: float = 0.01,
) -> torch.Tensor:
    """Focal loss with an added entropy penalty to prefer peaky outputs."""
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    with torch.no_grad():
        pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    mask = targets_flat != -1
    focal_mean = focal[mask].mean()

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy_mean = entropy[targets != -1].mean()
    return focal_mean + beta * entropy_mean


def rank_distance_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 1.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Rank-distance scaled focal loss to emphasize hard, misranked targets."""
    b, t, v = logits.shape
    logits_flat = logits.view(-1, v)
    targets_flat = targets.view(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-1)
    mask = targets_flat != -1
    with torch.no_grad():
        logits_sel = logits_flat[mask]
        targets_sel = targets_flat[mask]
        target_logits = logits_sel[torch.arange(logits_sel.size(0)), targets_sel]
        rank = (logits_sel > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
        rank_scale = 1 + gamma * (rank.float() - 1)
        rank_scale = torch.nan_to_num(rank_scale, nan=0.0, posinf=1e4, neginf=0.0)
        rank_scale = rank_scale.clamp(max=1e4)
        pt = torch.exp(-ce[mask])
        focal_scale = (1 - pt) ** focal_gamma
    scaled = torch.zeros_like(ce)
    scaled[mask] = ce[mask] * rank_scale * focal_scale
    return scaled[mask].mean()


def entropy_rank_distance_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    iter_num: int | None = None,
    gamma: float = 1.0,
    focal_gamma: float = 2.0,
    beta: float = 0.01,
) -> torch.Tensor:
    """Combine rank-distance scaling, focal weighting, and entropy penalty."""
    loss = rank_distance_focal_loss(
        logits, targets, iter_num=iter_num, gamma=gamma, focal_gamma=focal_gamma
    )
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    mask = targets != -1
    return loss + beta * entropy[mask].mean()


LOSS_VARIANTS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "cross_entropy": cross_entropy_loss,
    "label_smoothing": label_smoothing_loss,
    "focal": focal_loss,
    "top1_focus": top1_focus_loss,
    "top1_margin": top1_margin_loss,
    "entropy_penalty": entropy_penalty_loss,
    "top1_ratio": top1_ratio_loss,
    "rank_distance": rank_distance_loss,
    "flatness_boost": flatness_boost_loss,
    "entropy_focal": entropy_focal_loss,
    "rank_distance_focal": rank_distance_focal_loss,
    "entropy_rank_distance_focal": entropy_rank_distance_focal_loss,
}


# ---------------------------------------------------------------------------
# Loss scheduling
# ---------------------------------------------------------------------------


@dataclass
class ScheduledValue:
    """Schedule a scalar value over training iterations."""

    schedule: List[Tuple[int, float]]

    def __post_init__(self) -> None:
        self.schedule.sort(key=lambda x: x[0])

    def __call__(self, iter_num: int | None) -> float:
        val = self.schedule[0][1]
        if iter_num is not None:
            for step, candidate in self.schedule:
                if iter_num >= step:
                    val = candidate
                else:
                    break
        return val

@dataclass
class ScheduledLoss:
    """Switch between different loss functions at predefined iterations."""

    schedule: List[Tuple[int, str]]
    loss_dict: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

    def __post_init__(self) -> None:
        self.schedule.sort(key=lambda x: x[0])

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, *, iter_num: int | None = None) -> torch.Tensor:
        name = self.schedule[0][1]
        if iter_num is not None:
            for step, candidate in self.schedule:
                if iter_num >= step:
                    name = candidate
                else:
                    break
        return self.loss_dict[name](logits, targets, iter_num=iter_num)


def parse_loss_schedule(schedule_str: str) -> List[Tuple[int, str]]:
    """Parse a schedule string like ``"0:cross_entropy,1000:top1_focus"``."""
    schedule: List[Tuple[int, str]] = []
    for part in schedule_str.split(","):
        step_str, name = part.split(":")
        schedule.append((int(step_str), name.strip()))
    return schedule


def parse_value_schedule(schedule_str: str) -> ScheduledValue:
    """Parse a schedule string like ``"0:1.0,1000:2.0"`` for scalar values."""
    schedule: List[Tuple[int, float]] = []
    for part in schedule_str.split(","):
        step_str, value_str = part.split(":")
        schedule.append((int(step_str), float(value_str)))
    return ScheduledValue(schedule)


def build_loss_function(args) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the loss function or a scheduler based on ``args``."""
    schedule_str = getattr(args, "loss_schedule", None)
    if schedule_str:
        schedule = parse_loss_schedule(schedule_str)
        return ScheduledLoss(schedule, LOSS_VARIANTS)
    loss_name = getattr(args, "loss_fn", "cross_entropy")
    if "rank_distance" in loss_name:
        base = getattr(args, "rank_scale", 1.0)
        scale_sched_str = getattr(args, "rank_scale_schedule", None)
        scaler = parse_value_schedule(scale_sched_str) if scale_sched_str else None

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor, *, iter_num: int | None = None) -> torch.Tensor:
            gamma = scaler(iter_num) if scaler else base
            return LOSS_VARIANTS[loss_name](logits, targets, iter_num=iter_num, gamma=gamma)

        return loss_fn

    return LOSS_VARIANTS.get(loss_name, cross_entropy_loss)

