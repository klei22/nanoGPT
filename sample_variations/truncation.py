from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F


def _apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None:
        return logits
    k = min(top_k, logits.size(-1))
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = -float("inf")
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p <= 0 or top_p >= 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 0] = False

    unsorted_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
    unsorted_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    logits[unsorted_mask] = -float("inf")
    return logits


def _apply_min_p(logits: torch.Tensor, min_p: Optional[float]) -> torch.Tensor:
    if min_p is None or min_p <= 0:
        return logits

    probs = F.softmax(logits, dim=-1)
    threshold = probs.max(dim=-1, keepdim=True).values * min_p
    logits[probs < threshold] = -float("inf")
    return logits


def _apply_top_a(logits: torch.Tensor, top_a: Optional[float]) -> torch.Tensor:
    if top_a is None or top_a <= 0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1, keepdim=True).values
    threshold = (max_probs**2) * top_a
    logits[probs < threshold] = -float("inf")
    return logits


def _apply_epsilon_cutoff(logits: torch.Tensor, epsilon_cutoff: Optional[float]) -> torch.Tensor:
    if epsilon_cutoff is None or epsilon_cutoff <= 0:
        return logits

    probs = F.softmax(logits, dim=-1)
    mask = probs < epsilon_cutoff
    top_idx = probs.argmax(dim=-1, keepdim=True)
    mask.scatter_(dim=-1, index=top_idx, value=False)
    logits[mask] = -float("inf")
    return logits


def apply_probability_filters(
    logits: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    top_a: Optional[float] = None,
    epsilon_cutoff: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Apply truncation-style token filters in a deterministic order.

    Order: top-k -> top-p -> min-p -> top-a -> epsilon cutoff -> softmax-threshold.
    """

    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    logits = _apply_min_p(logits, min_p)
    logits = _apply_top_a(logits, top_a)
    logits = _apply_epsilon_cutoff(logits, epsilon_cutoff)

    if softmax_threshold is not None:
        probs = F.softmax(logits, dim=-1)
        threshold = probs.max(dim=-1, keepdim=True).values * softmax_threshold
        probs[probs < threshold] = 0
        return probs

    return F.softmax(logits, dim=-1)
