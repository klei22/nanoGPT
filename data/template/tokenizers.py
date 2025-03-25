# tokenizers.py
import os
import pickle
import tempfile
import sentencepiece as spm
import tiktoken
from tqdm import tqdm
from collections import defaultdict


class Tokenizer:
    def __init__(self, args):
        self.args = args
        self.token_counts = defaultdict(int) if getattr(args, "track_token_counts", False) else None

    def tokenize(self, data):
        raise NotImplementedError("Tokenize method must be implemented by subclasses.")

    def detokenize(self, ids):
        raise NotImplementedError("Detokenize method must be implemented by subclasses.")

    def save_meta(self, meta):
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def record_token(self, token_id):
        if self.token_counts is not None:
            self.token_counts[token_id] += 1

    def finalize_meta(self, meta):
        if self.token_counts is not None:
            meta["token_counts"] = dict(self.token_counts)
        self.save_meta(meta)

    @staticmethod
    def get_key_from_meta(keyname):
        meta_path = 'meta.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                return meta.get(keyname)
        return None


class NumericRangeTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        self.min_token = args.min_token
        self.max_token = args.max_token
        self.stoi = None
        self.itos = None

    def tokenize(self, data):
        tokens = []
        encountered_tokens = set()
        lines = data.strip().split('\n')
        for line in tqdm(lines, desc="Tokenizing Numeric Range"):
            try:
                num = int(line)
                if self.min_token <= num <= self.max_token:
                    tokens.append(num)
                    encountered_tokens.add(num)
                else:
                    print(f"Warning: Number {num} is outside the specified range and will be skipped.")
            except ValueError:
                print(f"Warning: Invalid number '{line}' will be skipped.")

        all_tokens = list(range(self.max_token, -1, -1))
        self.stoi = {str(num): i for i, num in enumerate(all_tokens)}
        self.itos = {i: str(num) for i, num in enumerate(all_tokens)}

        indexed_tokens = []
        for token in tokens:
            idx = self.stoi[str(token)]
            self.record_token(idx)
            indexed_tokens.append(idx)

        meta = {
            "vocab_size": len(self.stoi),
            "tokenizer": "numeric_range",
            "min_token": self.min_token,
            "max_token": self.max_token,
            "stoi": self.stoi,
            "itos": self.itos,
            "encountered_tokens": sorted(encountered_tokens, reverse=True)
        }
        self.finalize_meta(meta)
        return indexed_tokens

    def detokenize(self, ids):
        return '\n'.join([self.itos[id] for id in ids])


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, args, input_files=None):
        super().__init__(args)
        self.vocab_size = args.vocab_size
        self.spm_model_file = args.spm_model_file
        self.spm_vocab_file = args.spm_vocab_file
        self.skip_tokenization = args.skip_tokenization
        self.input_files = input_files
        self.sp = None

        if self.spm_model_file:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.spm_model_file)
        elif input_files:
            self.sp = self.train_sentencepiece_model()

    def train_sentencepiece_model(self):
        spm_model_prefix = "trained_spm_model"
        num_threads = os.cpu_count()
        input_arg = ""
        if isinstance(self.input_files, list):
            with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
                for input_file in self.input_files:
                    with open(input_file, "r") as infile:
                        tmpfile.write(infile.read())
                input_arg = tmpfile.name
        else:
            input_arg = self.input_files

        spm.SentencePieceTrainer.train(
            num_threads=num_threads,
            user_defined_symbols="\n, ",
            input=input_arg,
            model_prefix=spm_model_prefix,
            split_digits=True,
            vocab_size=self.vocab_size,
            model_type="bpe",
        )
        print("SentencePiece model training complete.")

        if isinstance(self.input_files, list):
            os.remove(input_arg)

        sp = spm.SentencePieceProcessor()
        sp.load(f"{spm_model_prefix}.model")
        return sp

    def tokenize(self, data):
        if not self.sp:
            raise ValueError("SentencePiece model is not loaded.")
        ids = self.sp.encode_as_ids(data)

        # Record token counts
        for token_id in ids:
            self.record_token(token_id)

        stoi = {self.sp.id_to_piece(i): i for i in range(self.sp.GetPieceSize())}
        itos = {i: self.sp.id_to_piece(i) for i in range(self.sp.GetPieceSize())}

        meta = {
            "vocab_size": self.sp.GetPieceSize(),
            "tokenizer": "sentencepiece",
            "stoi": stoi,
            "itos": itos,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        if not self.sp:
            raise ValueError("SentencePiece model is not loaded.")
        return self.sp.decode_ids(ids)

class TiktokenTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        self.tiktoken_encoding = args.tiktoken_encoding
        self.enc = tiktoken.get_encoding(self.tiktoken_encoding)
        self.vocab_size = self.enc.n_vocab

    def tokenize(self, data):
        ids = self.enc.encode_ordinary(data)
        for token_id in ids:
            self.record_token(token_id)
        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "tiktoken",
            "tiktoken_encoding": self.tiktoken_encoding,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        return self.enc.decode(ids)


class CustomTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        if args.tokens_file is None:
            raise ValueError("Tokens file must be provided for custom tokenization method.")
        with open(args.tokens_file, "r") as f:
            self.tokens = [line.strip() for line in f.readlines() if line.strip()]
            self.tokens = [token.replace("\\n", "\n").replace("\\t", "\t") for token in self.tokens]
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for i, token in enumerate(self.tokens)}

    def tokenize(self, data):
        encoded_data = []
        i = 0
        covered_chars = 0
        data_len = len(data)
        pbar = tqdm(total=data_len, desc="Tokenizing Custom Tokens")
        while i < data_len:
            matched = False
            for token in self.tokens:
                token_len = len(token)
                if data.startswith(token, i):
                    encoded_data.append(self.stoi[token])
                    self.record_token(self.stoi[token])
                    i += token_len
                    covered_chars += token_len
                    pbar.update(token_len)
                    matched = True
                    break
            if not matched:
                i += 1  # Skip character if no token matches
                pbar.update(1)
        pbar.close()
        coverage = covered_chars / data_len
        print(f"Data coverage by tokens: {coverage*100:.2f}%")
        meta = {"vocab_size": len(self.tokens), "stoi": self.stoi, "itos": self.itos}
        self.finalize_meta(meta)
        return encoded_data

    def detokenize(self, ids):
        return ''.join([self.itos[id] for id in ids])

class CharTokenizer(Tokenizer):
    def __init__(self, args, train_data, val_data):
        super().__init__(args)
        self.reuse_chars = args.reuse_chars
        if self.reuse_chars:
            self.chars = self.get_key_from_meta('chars')
            if self.chars is None:
                raise ValueError("No chars found in meta.pkl. Cannot reuse chars.")
        else:
            self.chars = sorted(list(set(train_data + (val_data if val_data else ""))))
            print(f"All unique characters: {''.join(self.chars)}")
            print(f"Vocab size: {len(self.chars)}")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def tokenize(self, data):
        data_len = len(data)
        ids = []
        pbar = tqdm(total=data_len, desc="Tokenizing Characters")
        for ch in data:
            token_id = self.stoi[ch]
            self.record_token(token_id)
            ids.append(token_id)
            pbar.update(1)

        pbar.close()
        meta = {"vocab_size": len(self.chars), "itos": self.itos, "stoi": self.stoi, "chars": self.chars}
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        return ''.join([self.itos[id] for id in ids])

class CustomCharTokenizerWithByteFallback(Tokenizer):
    """
    In this version, we assign IDs 0..255 to raw bytes,
    then custom tokens get IDs from 256 upwards.

    During tokenization:
      1) Convert text to UTF-8 bytes.
      2) For each position in the byte sequence, attempt to match
         a custom token's UTF-8 pattern. If we match, produce that token ID.
         Otherwise, produce the ID for the single byte.

    Detokenization:
      - If ID < 256, it's a single raw byte.
      - If ID >= 256, it's the custom token string.
    """

    def __init__(self, args):
        super().__init__(args)
        if args.custom_chars_file is None:
            raise ValueError("Custom characters file must be provided for this tokenizer.")

        # Load custom tokens from file
        with open(args.custom_chars_file, "r", encoding="utf-8") as f:
            self.custom_tokens = [line.strip() for line in f if line.strip()]

        # Build vocab dictionaries (bytes first, then custom tokens)
        self.build_vocab()

    def build_vocab(self):
        # Assign IDs 0..255 to individual bytes
        self.stoi = {}
        self.itos = {}

        for b in range(256):
            # Store key as the actual single byte
            key = bytes([b])
            self.stoi[key] = b  # ID = b
            self.itos[b] = key

        # Now assign IDs to the custom tokens from 256 onwards
        offset = 256
        self.custom_token_bytes = {}
        for i, token_str in enumerate(self.custom_tokens):
            token_id = offset + i
            self.stoi[token_str] = token_id
            self.itos[token_id] = token_str
            self.custom_token_bytes[token_str] = token_str.encode('utf-8')

        self.custom_char_count = len(self.custom_tokens)
        self.vocab_size = 256 + self.custom_char_count

    def tokenize(self, data):
        # Convert entire string to UTF-8 bytes
        data_bytes = data.encode('utf-8')
        i = 0
        n = len(data_bytes)
        ids = []

        # We'll try to match any custom token at the current position; otherwise single byte
        pbar = tqdm(total=n, desc="Tokenizing Bytes First + Custom")
        while i < n:
            matched = False
            # Check each custom token
            for token_str, token_bytes in self.custom_token_bytes.items():
                length = len(token_bytes)
                # If next 'length' bytes match this custom token
                if data_bytes[i:i+length] == token_bytes:
                    token_id = self.stoi[token_str]  # e.g., 256+
                    self.record_token(token_id)
                    ids.append(token_id)
                    i += length
                    pbar.update(length)
                    matched = True
                    break

            if not matched:
                # No custom token matched, so we treat this as a single byte
                single_byte = data_bytes[i:i+1]
                token_id = self.stoi[single_byte]  # 0..255
                self.record_token(token_id)
                ids.append(token_id)
                i += 1
                pbar.update(1)

        pbar.close()

        # Finalize metadata with token_counts
        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "custom_char_with_byte_fallback",
            "custom_chars": self.custom_tokens,  # i.e., custom tokens
            "stoi": self.stoi,
            "itos": self.itos,
            "custom_char_count": self.custom_char_count,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        """
        If ID < 256 => single byte
        If ID >= 256 => custom token string
        We'll accumulate bytes in a buffer, and whenever we see a custom token,
        we flush the buffer as text, then append the custom token as is.
        """
        out_pieces = []
        byte_buffer = []

        for idx, token_id in enumerate(ids):
            if token_id < 256:
                # Single raw byte
                byte_buffer.append(self.itos[token_id])  # e.g. b'\x61'
            else:
                # It's a custom token
                # First flush any accumulated bytes
                if byte_buffer:
                    all_bytes = b''.join(byte_buffer)
                    out_pieces.append(all_bytes.decode('utf-8', errors='replace'))
                    byte_buffer = []
                # Append the custom token string
                custom_str = self.itos[token_id]
                out_pieces.append(custom_str)

        # Flush remaining bytes
        if byte_buffer:
            all_bytes = b''.join(byte_buffer)
            out_pieces.append(all_bytes.decode('utf-8', errors='replace'))

        return ''.join(out_pieces)

