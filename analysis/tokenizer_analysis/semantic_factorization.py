#!/usr/bin/env python3
"""Estimate vocabulary savings from factoring orthographic token features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPACE_MARKER = "▁"


@dataclass(frozen=True)
class FactorizationResult:
    vocabulary_size: int
    leading_space: frozenset[str]
    initial_capital: frozenset[str]
    all_caps: frozenset[str]

    @property
    def removed(self) -> frozenset[str]:
        return self.leading_space | self.initial_capital | self.all_caps

    @property
    def factorized_vocabulary_size(self) -> int:
        return self.vocabulary_size - len(self.removed)

    def exclusive_counts(self) -> tuple[int, int, int]:
        """Return counts attributed in the requested space/capital/caps order."""
        capital = self.initial_capital - self.leading_space
        all_caps = self.all_caps - self.leading_space - self.initial_capital
        return len(self.leading_space), len(capital), len(all_caps)


def _split_space_marker(token: str) -> tuple[str, str]:
    return (SPACE_MARKER, token[1:]) if token.startswith(SPACE_MARKER) else ("", token)


def _has_case(character: str) -> bool:
    return character.lower() != character.upper()


def analyze_tokens(tokens: Iterable[str]) -> FactorizationResult:
    """Find tokens having exact base-token counterparts under three rules.

    Capitalization rules inspect the first decoded non-space character. This
    makes the features composable: ``▁Hello`` may carry SPACE and CAPITALIZE.
    Unicode's ``str.lower`` and ``str.isupper`` semantics are used.
    """
    vocabulary = frozenset(tokens)
    leading_space: set[str] = set()
    initial_capital: set[str] = set()
    all_caps: set[str] = set()

    for token in vocabulary:
        if token.startswith(SPACE_MARKER) and len(token) > 1 and token[1:] in vocabulary:
            leading_space.add(token)

        prefix, text = _split_space_marker(token)
        if text and text[0].isupper():
            counterpart = prefix + text[0].lower() + text[1:]
            if counterpart != token and counterpart in vocabulary:
                initial_capital.add(token)

        cased_characters = [character for character in text if _has_case(character)]
        if cased_characters and all(character.isupper() for character in cased_characters):
            counterpart = prefix + text.lower()
            if counterpart != token and counterpart in vocabulary:
                all_caps.add(token)

    return FactorizationResult(
        vocabulary_size=len(vocabulary),
        leading_space=frozenset(leading_space),
        initial_capital=frozenset(initial_capital),
        all_caps=frozenset(all_caps),
    )


def load_tokenizer_vocabulary(path: Path) -> dict[str, int]:
    """Load the BPE vocabulary from a Hugging Face ``tokenizer.json`` file."""
    with path.open(encoding="utf-8") as tokenizer_file:
        document = json.load(tokenizer_file)
    vocabulary = document.get("model", {}).get("vocab")
    if not isinstance(vocabulary, dict):
        raise ValueError(f"{path} does not contain an object at model.vocab")
    if not all(isinstance(token, str) and isinstance(token_id, int) for token, token_id in vocabulary.items()):
        raise ValueError(f"{path} has an invalid model.vocab")
    return vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json", type=Path, help="Downloaded Hugging Face tokenizer.json")
    parser.add_argument("--examples", type=int, default=5, help="Examples printed for each rule")
    args = parser.parse_args()

    vocabulary = load_tokenizer_vocabulary(args.tokenizer_json)
    result = analyze_tokens(vocabulary)
    exclusive = result.exclusive_counts()
    rules = (
        ("leading space", result.leading_space, exclusive[0]),
        ("initial capital", result.initial_capital, exclusive[1]),
        ("all caps", result.all_caps, exclusive[2]),
    )
    print(f"before: {result.vocabulary_size:,}")
    for label, matches, exclusive_count in rules:
        examples = sorted(matches, key=vocabulary.get)[: args.examples]
        print(f"{label}: {len(matches):,} matches; {exclusive_count:,} newly removed; examples={examples!r}")
    print(f"unique rows removed: {len(result.removed):,}")
    print(f"after: {result.factorized_vocabulary_size:,}")


if __name__ == "__main__":
    main()
