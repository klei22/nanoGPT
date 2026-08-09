#!/usr/bin/env python3
"""Export nanoGPT tokenizer metadata for Android assets.

The training loop copies data/<dataset>/meta.pkl into the checkpoint output
folder. This helper reads that meta.pkl and writes a JSON asset that Android can
load alongside the exported mobile model artifact.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping


CHAR_TOKENIZER_NAMES = {
    "char",
    "character",
    "character_level",
    "char_level",
    "char_bpe",
    "custom_char",
    "custom_char_with_byte_fallback",
    "json_byte_fallback",
}


def _json_key(value: Any) -> str:
    """Return a deterministic JSON object key for a tokenizer symbol."""
    if isinstance(value, bytes):
        # Keep byte fallback tokens representable in JSON while making their
        # special handling explicit to Android implementers.
        return "byte:" + value.hex()
    return str(value)


def _json_token(value: Any) -> str:
    """Return a JSON string for a tokenizer symbol value."""
    if isinstance(value, bytes):
        return "byte:" + value.hex()
    return str(value)


def _normalize_stoi(stoi: Mapping[Any, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for token, idx in stoi.items():
        normalized[_json_key(token)] = int(idx)
    return dict(sorted(normalized.items(), key=lambda item: item[1]))


def _normalize_itos(itos: Mapping[Any, Any] | list[Any] | tuple[Any, ...]) -> dict[str, str]:
    if isinstance(itos, Mapping):
        items = itos.items()
    else:
        items = enumerate(itos)
    normalized = {str(int(idx)): _json_token(token) for idx, token in items}
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def _infer_tokenizer_type(meta: Mapping[str, Any]) -> str:
    tokenizer = meta.get("tokenizer")
    if tokenizer is None and {"stoi", "itos"}.issubset(meta):
        return "char"
    if isinstance(tokenizer, str):
        return tokenizer
    return str(tokenizer)


def _is_json_exportable_char_tokenizer(meta: Mapping[str, Any], tokenizer_type: str) -> bool:
    if "stoi" not in meta or "itos" not in meta:
        return False
    return tokenizer_type in CHAR_TOKENIZER_NAMES or tokenizer_type not in {
        "tiktoken",
        "huggingface",
        "sentencepiece",
        "sinewave",
    }


def load_meta(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, Mapping):
        raise TypeError(f"Expected {path} to contain a dict-like object, got {type(meta)!r}")
    return meta


def build_tokenizer_json(meta: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer_type = _infer_tokenizer_type(meta)

    if tokenizer_type == "tiktoken":
        encoding = meta.get("tiktoken_encoding", "gpt2")
        raise ValueError(
            "meta.pkl declares a tiktoken tokenizer. Do not export stoi/itos JSON. "
            f"Use an Android-compatible {encoding!r} BPE/tiktoken implementation and "
            "ship its vocab/merge/special-token assets instead. See android_export/README.md."
        )

    if not _is_json_exportable_char_tokenizer(meta, tokenizer_type):
        raise ValueError(
            f"Tokenizer {tokenizer_type!r} is not exportable as char-level stoi/itos JSON. "
            "See android_export/README.md for non-char tokenizer asset requirements."
        )

    stoi = _normalize_stoi(meta["stoi"])
    itos = _normalize_itos(meta["itos"])
    vocab_size = int(meta.get("vocab_size", len(stoi)))

    if vocab_size != len(stoi) or len(stoi) != len(itos):
        raise ValueError(
            "Tokenizer metadata is inconsistent: "
            f"vocab_size={vocab_size}, len(stoi)={len(stoi)}, len(itos)={len(itos)}"
        )

    return {
        "tokenizer_type": tokenizer_type,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export checkpoint meta.pkl tokenizer metadata to an Android JSON asset."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("out"),
        help="Checkpoint output directory that should contain meta.pkl (default: out).",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Explicit meta.pkl path. Overrides --checkpoint-dir/meta.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("android_app/app/src/main/assets/tokenizer.json"),
        help="Output tokenizer JSON asset path.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON for review. The default is compact for assets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    meta_path = args.meta_path or args.checkpoint_dir / "meta.pkl"

    if not meta_path.exists():
        print(
            f"error: meta.pkl not found at {meta_path}. "
            "Train/export should copy data/<dataset>/meta.pkl into the checkpoint output directory, "
            "or pass --meta-path explicitly.",
            file=sys.stderr,
        )
        return 1

    meta = load_meta(meta_path)
    exported = build_tokenizer_json(meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            exported,
            f,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
        )
        f.write("\n")

    print(f"wrote {args.output} from {meta_path}")
    print(f"tokenizer_type={exported['tokenizer_type']} vocab_size={exported['vocab_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
