import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

try:
    import sacrebleu  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sacrebleu = None


@dataclass
class BenchmarkCase:
    """Single benchmark example containing the prompt and expected answer."""

    start_tokens: str
    answer_string: Optional[str] = None
    legal_answer_regex: Optional[str] = None
    references: Optional[List[str]] = None


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark suite, typically loaded from JSON."""

    tasks: List[BenchmarkCase]
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_k: Sequence[Optional[int]] = field(default_factory=lambda: [None])
    scorers: Sequence[str] = field(default_factory=lambda: ("exact", "legal"))

    @staticmethod
    def from_file(path: str) -> "BenchmarkConfig":
        with open(path, "r") as f:
            raw = json.load(f)

        tasks = [BenchmarkCase(**t) for t in raw.get("tasks", [])]
        if not tasks:
            raise ValueError("Benchmark config must contain at least one task entry")

        return BenchmarkConfig(
            tasks=tasks,
            max_new_tokens=raw.get("max_new_tokens", 16),
            temperature=raw.get("temperature", 0.0),
            top_k=raw.get("top_k", [None]),
            scorers=raw.get("scorers", ["exact", "legal"]),
        )


def _strip_prefix(full_text: str, prefix: str) -> str:
    if full_text.startswith(prefix):
        return full_text[len(prefix) :]
    return full_text


def _take_until_newline(text: str) -> str:
    newline_idx = text.find("\n")
    return text if newline_idx == -1 else text[:newline_idx]


def exact_match(prediction: str, case: BenchmarkCase) -> float:
    if case.answer_string is None:
        return 0.0
    return float(prediction.strip() == case.answer_string.strip())


def legal_match(prediction: str, case: BenchmarkCase) -> float:
    if not case.legal_answer_regex:
        return 0.0
    return float(re.fullmatch(case.legal_answer_regex, prediction.strip()) is not None)


def bleu_sentence(prediction: str, case: BenchmarkCase) -> float:
    if sacrebleu is None or not (case.references or case.answer_string):
        return 0.0
    refs = case.references or [case.answer_string]
    return sacrebleu.sentence_bleu(prediction, refs).score


SCORER_REGISTRY: Dict[str, Callable[[str, BenchmarkCase], float]] = {
    "exact": exact_match,
    "legal": legal_match,
    "bleu": bleu_sentence,
}


class InferenceBenchmarkRunner:
    """Run generation-time benchmarks in a modular way."""

    def __init__(
        self,
        model,
        encode: Callable[[str], List[int]],
        decode: Callable[[List[int]], str],
        device: torch.device,
        default_max_new_tokens: int,
        default_temperature: float,
        top_k_values: Sequence[Optional[int]],
    ) -> None:
        self.model = model
        self.encode = encode
        self.decode = decode
        self.device = device
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.top_k_values = list(dict.fromkeys(top_k_values))  # deduplicate while preserving order

    def run(self, config: BenchmarkConfig) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        max_new_tokens = config.max_new_tokens or self.default_max_new_tokens
        temperature = config.temperature if config.temperature is not None else self.default_temperature

        for top_k in config.top_k or self.top_k_values:
            k_label = self._format_top_k(top_k)
            metrics: Dict[str, List[float]] = {name: [] for name in config.scorers}

            for case in config.tasks:
                generated = self._generate(case.start_tokens, max_new_tokens, temperature, top_k)
                answer_fragment = _take_until_newline(_strip_prefix(generated, case.start_tokens))

                for scorer_name in config.scorers:
                    scorer = SCORER_REGISTRY.get(scorer_name)
                    if scorer is None:
                        continue
                    metrics[scorer_name].append(scorer(answer_fragment, case))

            results[k_label] = {
                name: float(sum(vals) / len(vals)) if vals else math.nan for name, vals in metrics.items()
            }

        return results

    def _generate(
        self,
        start_tokens: str,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
    ) -> str:
        start_ids = torch.tensor(self.encode(start_tokens), dtype=torch.long, device=self.device)[None, ...]
        with torch.no_grad():
            y = self.model.generate(start_ids, max_new_tokens, temperature=temperature, top_k=top_k)
        return self.decode(y[0].tolist())

    @staticmethod
    def _format_top_k(top_k: Optional[int]) -> str:
        return "top_k_none" if top_k is None else f"top_k_{top_k}"

