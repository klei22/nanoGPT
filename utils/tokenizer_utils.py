"""Utilities for working with tokenizer metadata.

These helpers were originally implemented in :mod:`sample.py` but are
needed in multiple modules (training, sampling, benchmarking).  They
provide encode/decode functions for the different tokenizer flavours that
can be described by ``meta.pkl`` files.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import tiktoken


def custom_char_with_byte_fallback_encode(text: str, stoi: Dict[object, int]) -> List[int]:
    """Encode ``text`` using a byte-level vocabulary with optional custom tokens.

    This mirrors the logic in ``CustomCharTokenizerWithByteFallback``. For each
    position in the UTF-8 byte stream we try each custom token (in the order
    they were defined) and fall back to emitting the raw byte ID when none
    match.
    """

    custom_token_bytes: List[Tuple[str, bytes]] = [
        (token, token.encode("utf-8"))
        for token in stoi.keys()
        if isinstance(token, str)
    ]

    data_bytes = text.encode("utf-8")
    i, n = 0, len(data_bytes)
    ids: List[int] = []

    while i < n:
        matched = False
        for token_str, token_bytes in custom_token_bytes:
            length = len(token_bytes)
            if data_bytes[i : i + length] == token_bytes:
                ids.append(stoi[token_str])
                i += length
                matched = True
                break
        if not matched:
            byte_token = data_bytes[i : i + 1]
            ids.append(stoi[byte_token])
            i += 1

    return ids


def custom_char_with_byte_fallback_decode(ids: Iterable[int], itos: Dict[int, object]) -> str:
    """Decode a list of token IDs produced by the byte-fallback tokenizer."""

    out_parts: List[str] = []
    byte_buffer: List[bytes] = []

    def flush_bytes() -> None:
        if byte_buffer:
            out_parts.append(b"".join(byte_buffer).decode("utf-8", errors="replace"))
            byte_buffer.clear()

    for tok_id in ids:
        token = itos[tok_id]
        if isinstance(token, bytes) and len(token) == 1:
            byte_buffer.append(token)
        elif isinstance(token, int) and token < 256:
            byte_buffer.append(bytes([token]))
        else:
            flush_bytes()
            if isinstance(token, bytes):
                out_parts.append(token.decode("utf-8", errors="replace"))
            else:
                out_parts.append(str(token))

    flush_bytes()
    return "".join(out_parts)


def byte_encode(text: str) -> List[int]:
    """Encode text into raw UTF-8 byte values."""

    return list(text.encode("utf-8"))


def byte_decode(ids: Iterable[int]) -> str:
    """Decode a list of raw byte values back into text."""

    return bytes(ids).decode("utf-8", errors="replace")


def get_tokenizer_functions(meta: Dict[str, object]) -> Tuple[Callable[[str], List[int]], Callable[[Iterable[int]], str]]:
    """Get encode/decode functions based on tokenizer metadata."""

    if "tokenizer" not in meta:
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
        return encode, decode

    if meta["tokenizer"] == "sinewave":
        def encode_fn(s: str) -> List[int]:
            s = s.strip()
            if not s:
                return []
            return [int(v) for v in s.split(",")]

        def decode_fn(values: Iterable[int]) -> str:
            return ",".join(str(int(v)) for v in values)

        return encode_fn, decode_fn

    if meta["tokenizer"] == "tiktoken":
        enc = tiktoken.get_encoding(meta["tiktoken_encoding"])
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)
        return encode, decode

    if meta["tokenizer"] == "byte":
        return byte_encode, byte_decode

    if meta["tokenizer"] == "custom_char_with_byte_fallback":
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: custom_char_with_byte_fallback_encode(s, stoi)
        decode = lambda l: custom_char_with_byte_fallback_decode(l, itos)
        return encode, decode

    if meta["tokenizer"] == "json_byte_fallback":
        stoi, itos = meta["stoi"], meta["itos"]
        string_token_tuples = [
            (token, token_id)
            for token, token_id in stoi.items()
            if isinstance(token, str)
        ]
        string_token_tuples.sort(key=lambda item: len(item[0]), reverse=True)

        def encode(text: str) -> List[int]:
            ids: List[int] = []
            current_pos = 0
            text_len = len(text)

            while current_pos < text_len:
                remaining_text = text[current_pos:]
                token_found = False

                for token, token_id in string_token_tuples:
                    if remaining_text.startswith(token):
                        ids.append(token_id)
                        current_pos += len(token)
                        token_found = True
                        break

                if not token_found:
                    char = text[current_pos]
                    char_bytes = char.encode("utf-8")
                    for byte in char_bytes:
                        byte_token = bytes([byte])
                        if byte_token in stoi:
                            ids.append(stoi[byte_token])
                        else:
                            ids.append(stoi.get("<unk>", 0))
                    current_pos += 1

            return ids

        def decode(token_ids: Iterable[int]) -> str:
            tokens: List[str] = []
            byte_buffer: List[int] = []

            for token_id in token_ids:
                if token_id not in itos:
                    continue

                token = itos[token_id]
                if isinstance(token, bytes):
                    byte_buffer.append(token[0])
                else:
                    if byte_buffer:
                        tokens.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                        byte_buffer.clear()
                    tokens.append(token.replace("Ġ", " "))

            if byte_buffer:
                tokens.append(bytes(byte_buffer).decode("utf-8", errors="replace"))

            return "".join(tokens)

        return encode, decode

    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return encode, decode


__all__ = [
    "custom_char_with_byte_fallback_encode",
    "custom_char_with_byte_fallback_decode",
    "byte_encode",
    "byte_decode",
    "get_tokenizer_functions",
]
