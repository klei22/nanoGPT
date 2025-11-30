"""Utilities for computing inference score variations on generated outputs."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, MutableMapping, Optional, Sequence

import math

ScoreFunction = Callable[[str, Optional[str], Callable[[str], Sequence[int]], str], float]


def _truncate_at_stop(generated: str, stop_string: str) -> str:
    """Return ``generated`` truncated at the first occurrence of ``stop_string``."""
    if not stop_string:
        return generated
    stop_index = generated.find(stop_string)
    if stop_index == -1:
        return generated
    return generated[:stop_index]


def _matching_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    """Count the number of matching items from the start of ``a`` and ``b``."""
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def greedy_correct_percent(
    generated: str,
    target: Optional[str],
    encode: Callable[[str], Sequence[int]],
    stop_string: str,
) -> float:
    """Return the percentage of target tokens that match greedily from the start."""
    if not target:
        return math.nan

    truncated = _truncate_at_stop(generated, stop_string)
    target_tokens = encode(target)
    if not target_tokens:
        return math.nan

    generated_tokens = encode(truncated)
    if not generated_tokens:
        return 0.0

    correct = _matching_prefix_length(generated_tokens, target_tokens)
    return 100.0 * correct / len(target_tokens)


def net_token_diff(
    generated: str,
    target: Optional[str],
    encode: Callable[[str], Sequence[int]],
    stop_string: str,
) -> float:
    """Return (# of matching prefix tokens) - (# of non-matching tokens)."""
    if not target:
        return math.nan

    truncated = _truncate_at_stop(generated, stop_string)
    generated_tokens = encode(truncated)
    target_tokens = encode(target)
    if not generated_tokens:
        return -float(len(target_tokens)) if target_tokens else 0.0

    correct = _matching_prefix_length(generated_tokens, target_tokens)
    incorrect = max(len(generated_tokens) - correct, 0)
    return float(correct - incorrect)


_SCORE_VARIATIONS: Dict[str, ScoreFunction] = {
    "greedy_correct_pct": greedy_correct_percent,
    "net_token_diff": net_token_diff,
}


def available_score_variations() -> Sequence[str]:
    """Return the names of the registered score variations."""
    return tuple(_SCORE_VARIATIONS.keys())


def get_score_function(name: str) -> ScoreFunction:
    """Return the score function registered under ``name``."""
    try:
        return _SCORE_VARIATIONS[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        available = ", ".join(sorted(_SCORE_VARIATIONS))
        raise KeyError(f"Unknown score variation '{name}'. Available: {available}") from exc


def compute_scores(
    names: Iterable[str],
    generated: str,
    target: Optional[str],
    encode: Callable[[str], Sequence[int]],
    stop_string: str,
) -> Dict[str, float]:
    """Compute the requested score variations for the given generation."""
    scores: MutableMapping[str, float] = {}
    for name in names:
        func = get_score_function(name)
        scores[name] = func(generated, target, encode, stop_string)
    return dict(scores)


__all__ = [
    "available_score_variations",
    "compute_scores",
    "greedy_correct_percent",
    "net_token_diff",
]
