import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from tokenizers import (
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    ByteTokenizer,
    CharTokenizer,
    CharBPETokenizerWithByteFallback,
    CustomCharTokenizerWithByteFallback,
    JsonByteTokenizerWithByteFallback,
    PythonProgrammingTokenizer,
    SineWaveTokenizer,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tokenization metrics (compression, Rényi entropy, MorphScore, byte premium)."
    )
    parser.add_argument("--text", type=str, required=True, help="Path to input text file or directory.")
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "sentencepiece",
            "tiktoken",
            "char",
            "char_bpe",
            "custom",
            "byte",
            "custom_char_byte_fallback",
            "json_byte_fallback",
            "python_programming",
            "sinewave",
        ],
        default="tiktoken",
        help="Tokenization method.",
    )

    # SentencePiece
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size for SentencePiece model.")
    parser.add_argument("--spm_model_file", type=str, default=None, help="Path to the pre-trained SentencePiece model file.")
    parser.add_argument("--spm_vocab_file", type=str, default=None, help="Path to the SentencePiece vocabulary file.")

    # Tiktoken
    parser.add_argument(
        "-e",
        "--tiktoken_encoding",
        choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"],
        default="gpt2",
        help="Version of tiktoken encoding to utilize",
    )
    parser.add_argument(
        "--additional_tokens_file",
        type=str,
        default=None,
        help="Path to JSON file containing additional special tokens for tiktoken (format: {'token': id})",
    )

    # Custom tokenizers
    parser.add_argument("--tokens_file", type=str, default=None, help="Path to newline-separated tokens for custom tokenization")
    parser.add_argument("--custom_chars_file", type=str, default=None, help="Path to custom characters for byte fallback tokenizer")
    parser.add_argument("--json_tokens_file", type=str, default=None, help="Path to JSON file containing tokens for json_byte_fallback tokenizer")

    # Metrics options
    parser.add_argument(
        "--morph_data",
        type=str,
        default=None,
        help=(
            "Optional path to morphological boundary data (TSV or JSONL). "
            "TSV supports 'word<TAB>boundary_index' or 'word<TAB>left<TAB>right'. "
            "JSONL supports {'word': str, 'boundary_index': int} or {'word': str, 'left': str, 'right': str}."
        ),
    )
    parser.add_argument(
        "--morph_include_single_tokens",
        action="store_true",
        help="Include single-token words when computing MorphScore.",
    )
    parser.add_argument(
        "--renyi_alpha",
        type=float,
        default=2.5,
        help="Alpha parameter for Rényi entropy (default: 2.5).",
    )
    parser.add_argument(
        "--byte_premium_reference",
        type=str,
        default=None,
        help="Optional reference text file with matched content to compute byte premium ratio.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="tokenization_metrics.json",
        help="Output path for metrics JSON.",
    )
    return parser.parse_args()


def _read_input_data(path: str) -> str:
    if os.path.isdir(path):
        collected = []
        for root, _, files in os.walk(path):
            for name in sorted(files):
                file_path = os.path.join(root, name)
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    collected.append(f.read())
        return "\n".join(collected)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_tokenizer(args: argparse.Namespace, text: str):
    if args.method == "sentencepiece":
        return SentencePieceTokenizer(args, input_files=args.text)
    if args.method == "tiktoken":
        return TiktokenTokenizer(args)
    if args.method == "custom":
        return CustomTokenizer(args)
    if args.method == "byte":
        return ByteTokenizer(args)
    if args.method == "char":
        return CharTokenizer(args, text, None)
    if args.method == "char_bpe":
        return CharBPETokenizerWithByteFallback(args, text, None)
    if args.method == "custom_char_byte_fallback":
        return CustomCharTokenizerWithByteFallback(args)
    if args.method == "json_byte_fallback":
        return JsonByteTokenizerWithByteFallback(args)
    if args.method == "python_programming":
        return PythonProgrammingTokenizer(args)
    if args.method == "sinewave":
        return SineWaveTokenizer(args)
    raise ValueError(f"Unknown tokenization method: {args.method}")


def _token_pieces(tokenizer, token_ids: List[int]) -> List[str]:
    pieces: List[str] = []
    if isinstance(tokenizer, SentencePieceTokenizer):
        for token_id in token_ids:
            pieces.append(tokenizer.sp.IdToPiece(token_id))
        return pieces

    if isinstance(tokenizer, TiktokenTokenizer):
        for token_id in token_ids:
            token_text = None
            for token, special_id in tokenizer.special_tokens.items():
                if token_id == special_id:
                    token_text = token
                    break
            if token_text is None:
                token_text = tokenizer.enc.decode([token_id])
            pieces.append(token_text)
        return pieces

    if hasattr(tokenizer, "itos"):
        for token_id in token_ids:
            piece = tokenizer.itos.get(token_id, "")
            if isinstance(piece, bytes):
                piece = piece.decode("utf-8", errors="replace")
            pieces.append(piece)
        return pieces

    return [str(token_id) for token_id in token_ids]


def _normalize_piece(piece: str, is_first: bool) -> str:
    if not is_first:
        return piece
    stripped = piece.lstrip()
    if stripped.startswith("▁"):
        stripped = stripped.lstrip("▁")
    if stripped.startswith("Ġ"):
        stripped = stripped.lstrip("Ġ")
    return stripped


def _token_boundaries(word: str, token_pieces: List[str]) -> List[int]:
    boundaries = []
    cursor = 0
    for idx, piece in enumerate(token_pieces):
        normalized = _normalize_piece(piece, idx == 0)
        if not normalized:
            continue
        cursor += len(normalized)
        boundaries.append(cursor)
    if boundaries and boundaries[-1] == len(word):
        boundaries.pop()
    return boundaries


def _load_morph_items(path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                word = payload.get("word")
                if word is None:
                    continue
                if "boundary_index" in payload:
                    boundary_index = int(payload["boundary_index"])
                elif "left" in payload and "right" in payload:
                    boundary_index = len(payload["left"])
                else:
                    continue
                items.append((word, boundary_index))
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                word, boundary_str = parts
                items.append((word, int(boundary_str)))
            elif len(parts) >= 3:
                word, left, _right = parts[:3]
                items.append((word, len(left)))
    return items


def _encode_for_morphscore(tokenizer, word: str) -> List[int]:
    if isinstance(tokenizer, SentencePieceTokenizer):
        return tokenizer.sp.encode_as_ids(word)
    if isinstance(tokenizer, TiktokenTokenizer):
        return tokenizer.enc.encode(word, allowed_special=set(), disallowed_special=())
    if isinstance(tokenizer, ByteTokenizer):
        return list(word.encode("utf-8"))
    if isinstance(tokenizer, CharTokenizer):
        return [tokenizer.stoi[ch] for ch in word]
    if isinstance(tokenizer, CustomTokenizer):
        ids = []
        i = 0
        data_len = len(word)
        while i < data_len:
            matched = False
            for token in tokenizer.tokens:
                token_len = len(token)
                if word.startswith(token, i):
                    ids.append(tokenizer.stoi[token])
                    i += token_len
                    matched = True
                    break
            if not matched:
                i += 1
        return ids
    if isinstance(tokenizer, CharBPETokenizerWithByteFallback):
        return tokenizer.tokenize(word)
    if isinstance(tokenizer, (CustomCharTokenizerWithByteFallback, JsonByteTokenizerWithByteFallback, PythonProgrammingTokenizer)):
        return tokenizer.tokenize(word)
    if isinstance(tokenizer, SineWaveTokenizer):
        return tokenizer.tokenize(word)
    return tokenizer.tokenize(word)


def compute_morphscore(
    tokenizer,
    items: Iterable[Tuple[str, int]],
    include_single_tokens: bool,
) -> Dict[str, float]:
    total = 0
    correct = 0
    skipped = 0

    for word, boundary_index in items:
        token_ids = _encode_for_morphscore(tokenizer, word)
        if len(token_ids) <= 1 and not include_single_tokens:
            skipped += 1
            continue
        pieces = _token_pieces(tokenizer, token_ids)
        boundaries = _token_boundaries(word, pieces)
        total += 1
        if boundary_index in boundaries:
            correct += 1

    score = (correct / total) if total else 0.0
    return {
        "morphscore": score,
        "morphscore_total": total,
        "morphscore_correct": correct,
        "morphscore_skipped": skipped,
    }


def compute_renyi_entropy(token_counts: Counter, alpha: float) -> float:
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    if alpha == 1.0:
        probs = [count / total for count in token_counts.values()]
        return -sum(p * math.log(p) for p in probs if p > 0)
    sum_prob = 0.0
    for count in token_counts.values():
        p_i = count / total
        sum_prob += p_i**alpha
    return (1 / (1 - alpha)) * math.log(sum_prob)


def main() -> None:
    args = parse_arguments()
    text = _read_input_data(args.text)

    tokenizer = _load_tokenizer(args, text)
    token_ids = tokenizer.tokenize(text)
    token_counts = Counter(token_ids)

    metrics: Dict[str, float] = {}
    metrics["token_count"] = len(token_ids)
    metrics["character_count"] = len(text)
    metrics["byte_count"] = len(text.encode("utf-8"))
    metrics["tokens_per_character"] = len(token_ids) / max(len(text), 1)
    metrics["tokens_per_byte"] = len(token_ids) / max(len(text.encode("utf-8")), 1)

    metrics["renyi_entropy"] = compute_renyi_entropy(token_counts, args.renyi_alpha)
    metrics["renyi_alpha"] = args.renyi_alpha

    if args.morph_data:
        items = _load_morph_items(args.morph_data)
        metrics.update(compute_morphscore(tokenizer, items, args.morph_include_single_tokens))

    if args.byte_premium_reference:
        reference_text = _read_input_data(args.byte_premium_reference)
        reference_bytes = len(reference_text.encode("utf-8"))
        metrics["byte_premium"] = metrics["byte_count"] / max(reference_bytes, 1)
        metrics["byte_premium_reference_bytes"] = reference_bytes

    output = {
        "method": args.method,
        "metrics": metrics,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
