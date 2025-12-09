import io
import json
import keyword
import os
import tokenize
from typing import List, Tuple

from tokenizers import Tokenizer


def _default_tokens_file() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "premade_vocab_sets", "python_programming_tokens.json")


class PythonProgrammingTokenizer(Tokenizer):
    """Tokenize Python source code with keyword tokens and byte fallback.

    * Reserved/control-flow tokens are loaded from a JSON vocabulary.
    * Comments and user-defined variable names are always emitted as raw bytes.
    * Keywords inside comments are treated as bytes because comment spans bypass
      the keyword/token matching logic.
    """

    def __init__(self, args):
        super().__init__(args)
        tokens_file = getattr(args, "python_tokens_file", None) or getattr(args, "json_tokens_file", None)
        self.tokens_file = tokens_file or _default_tokens_file()

        with open(self.tokens_file, "r", encoding="utf-8") as f:
            self.custom_tokens: List[str] = json.load(f)
            if not isinstance(self.custom_tokens, list):
                raise ValueError("PythonProgrammingTokenizer expects a JSON array of tokens")

        self._build_vocab()

    def _build_vocab(self) -> None:
        self.stoi = {}
        self.itos = {}

        for b in range(256):
            key = bytes([b])
            self.stoi[key] = b
            self.itos[b] = key

        offset = 256
        self.custom_token_bytes = {}
        for i, token_str in enumerate(self.custom_tokens):
            token_id = offset + i
            self.stoi[token_str] = token_id
            self.itos[token_id] = token_str
            self.custom_token_bytes[token_str] = token_str.encode("utf-8")

        self.vocab_size = 256 + len(self.custom_tokens)

    def _emit_bytes(self, text: str, ids: List[int]) -> None:
        for byte_val in text.encode("utf-8"):
            token_id = self.stoi[bytes([byte_val])]
            self.record_token(token_id)
            ids.append(token_id)

    def _emit_token(self, token_str: str, ids: List[int]) -> None:
        token_id = self.stoi.get(token_str)
        if token_id is None:
            self._emit_bytes(token_str, ids)
            return

        self.record_token(token_id)
        ids.append(token_id)

    @staticmethod
    def _line_offsets(text: str) -> List[int]:
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    @staticmethod
    def _pos_to_index(offsets: List[int], position: Tuple[int, int]) -> int:
        line, col = position
        return offsets[line - 1] + col

    def tokenize(self, data: str):
        ids: List[int] = []
        line_offsets = self._line_offsets(data)

        last_index = 0
        reader = io.StringIO(data).readline

        try:
            for tok in tokenize.generate_tokens(reader):
                start_idx = self._pos_to_index(line_offsets, tok.start)
                end_idx = self._pos_to_index(line_offsets, tok.end)

                if start_idx > last_index:
                    self._emit_bytes(data[last_index:start_idx], ids)

                if tok.type == tokenize.COMMENT:
                    self._emit_bytes(tok.string, ids)
                elif tok.type == tokenize.NAME:
                    if keyword.iskeyword(tok.string):
                        self._emit_token(tok.string, ids)
                    else:
                        self._emit_bytes(tok.string, ids)
                elif tok.string in self.custom_token_bytes:
                    self._emit_token(tok.string, ids)
                else:
                    self._emit_bytes(tok.string, ids)

                last_index = end_idx
        except tokenize.TokenError as e:
            error_pos = e.args[1] if len(e.args) > 1 else None
            try:
                error_index = self._pos_to_index(line_offsets, error_pos) if error_pos else len(data)
            except Exception:
                error_index = len(data)
            error_index = max(error_index, last_index)
            if error_index > last_index:
                self._emit_bytes(data[last_index:error_index], ids)
                last_index = error_index

        if last_index < len(data):
            self._emit_bytes(data[last_index:], ids)

        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "python_programming",
            "custom_tokens": self.custom_tokens,
            "stoi": self.stoi,
            "itos": self.itos,
            "tokens_file": os.path.abspath(self.tokens_file),
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids: List[int]) -> str:
        out_parts: List[str] = []
        byte_buffer: List[bytes] = []

        def flush_bytes():
            if byte_buffer:
                out_parts.append(b"".join(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id < 256:
                byte_buffer.append(self.itos[token_id])
            else:
                flush_bytes()
                out_parts.append(self.itos[token_id])

        flush_bytes()
        return "".join(out_parts)
