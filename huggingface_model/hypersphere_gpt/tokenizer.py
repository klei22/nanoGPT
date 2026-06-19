"""Utilities for working with the tiktoken tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - optional dependency for runtime
    raise RuntimeError(
        "tiktoken must be installed to use the Hypersphere GPT tokenizer utilities"
    ) from exc


@dataclass
class TiktokenTokenizer:
    """Thin wrapper that mimics the Hugging Face tokenizer API."""

    name: str = "gpt2"

    def __post_init__(self) -> None:
        self.encoding = tiktoken.get_encoding(self.name)
        self.eos_token_id = self.encoding.eot_token
        self.pad_token_id = self.eos_token_id
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> List[int]:
        ids = self.encoding.encode(text, allowed_special={""})
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.encoding.decode(ids)
