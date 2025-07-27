import re
import numpy as np
from spellchecker import SpellChecker

_spell = SpellChecker()


def spelling_correctness(text: str) -> float:
    """Return fraction of words spelled correctly using pyspellchecker."""
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    misspelled = _spell.unknown([w.lower() for w in words])
    return 1.0 - len(misspelled) / len(words)


def type_token_ratio(text: str) -> float:
    """Return vocabulary richness (unique words / total words)."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def average_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences if s.strip()]
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def run_benchmarks(text: str) -> dict:
    """Compute all dataset benchmarks for given text."""
    return {
        "spelling_correctness": spelling_correctness(text),
        "type_token_ratio": type_token_ratio(text),
        "average_sentence_length": average_sentence_length(text),
    }
