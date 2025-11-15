"""Utility to extract Hangul-only tokens from a SentencePiece vocabulary.

This script reads a JSON file containing a list of tokens (such as
``gemma_tokens.json``) and writes a new JSON file containing only the
tokens that are fully composed of Hangul code points.  The resulting
tokens are sorted from the longest (in number of Unicode code points) to
the shortest.

Usage
-----

```
python extract_hangul_tokens.py --input gemma_tokens.json --output hangul_tokens.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# Inclusive Unicode ranges that cover Hangul characters.  These ranges are
# based on the definitions from the Unicode standard.
HANGUL_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    (0xFFA0, 0xFFDC),  # Halfwidth Hangul variants
)


def _is_hangul_char(ch: str) -> bool:
    """Return ``True`` if *ch* lies within any of the Hangul ranges."""

    codepoint = ord(ch)
    return any(start <= codepoint <= end for start, end in HANGUL_RANGES)


def is_hangul_token(token: str) -> bool:
    """Return ``True`` if *token* is non-empty and every character is Hangul."""

    if not token:
        return False
    return all(_is_hangul_char(ch) for ch in token)


def extract_hangul_tokens(tokens: Sequence[str]) -> List[str]:
    """Return a list of Hangul-only tokens sorted by descending length."""

    hangul_tokens = [token for token in tokens if is_hangul_token(token)]
    hangul_tokens.sort(key=len, reverse=True)
    return hangul_tokens


def _load_tokens(path: Path) -> Sequence[str]:
    with path.open("r", encoding="utf-8") as fp:
        tokens = json.load(fp)
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError("Input JSON must be a list of strings.")
    return tokens


def _write_tokens(tokens: Iterable[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(list(tokens), fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source JSON file containing the token list.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the Hangul-only JSON token list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = _load_tokens(args.input)
    hangul_tokens = extract_hangul_tokens(tokens)
    _write_tokens(hangul_tokens, args.output)


if __name__ == "__main__":
    main()
