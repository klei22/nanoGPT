"""Utilities for tokenizing Python code with reserved tokens and byte fallback."""
from __future__ import annotations

import io
import tokenize
from typing import Callable, Iterable, Sequence


class PythonTokenProcessor:
    """Apply Python-aware tokenization while preserving raw text outside reserved tokens.

    The processor walks the Python token stream and only emits reserved tokens for
    actual code identifiers/operators. Everything else (comments, strings, variable
    names, whitespace, etc.) should be emitted through the provided byte encoder.
    """

    def __init__(self, reserved_tokens: Iterable[str]):
        self.reserved_tokens = set(reserved_tokens)

    def encode_with_reserved_tokens(
        self,
        source: str,
        encode_bytes: Callable[[str], None],
        emit_reserved: Callable[[str], None],
    ) -> None:
        """Encode *source* by mixing reserved tokens and byte-level segments.

        Args:
            source: Full Python source text.
            encode_bytes: Callback used for any substring that should be handled at
                the byte level.
            emit_reserved: Callback used when a reserved token should be emitted.
        """
        lines: Sequence[str] = source.splitlines(keepends=True)
        line_offsets = self._compute_line_offsets(lines)

        def to_index(position: tuple[int, int]) -> int:
            line_no, col = position
            return line_offsets[line_no - 1] + col

        reader = io.StringIO(source).readline
        last_index = 0

        for token in tokenize.generate_tokens(reader):
            if token.type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue

            start_idx = to_index(token.start)
            end_idx = to_index(token.end)

            if start_idx > last_index:
                encode_bytes(source[last_index:start_idx])

            token_text = source[start_idx:end_idx]

            if token.type == tokenize.COMMENT:
                encode_bytes(token_text)
            elif token.type == tokenize.NAME and token_text in self.reserved_tokens:
                emit_reserved(token_text)
            elif token.type == tokenize.OP and token_text in self.reserved_tokens:
                emit_reserved(token_text)
            else:
                encode_bytes(token_text)

            last_index = end_idx

        if last_index < len(source):
            encode_bytes(source[last_index:])

    @staticmethod
    def _compute_line_offsets(lines: Sequence[str]) -> list[int]:
        offsets: list[int] = []
        running_total = 0
        for line in lines:
            offsets.append(running_total)
            running_total += len(line)
        if not offsets:
            offsets.append(0)
        return offsets
