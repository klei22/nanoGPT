#!/usr/bin/env python3
"""Prepare an audio numerical multicontext dataset from tokenized audio text."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _read_input_data(path):
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


def _resolve_output_path(output_dir, path):
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return Path(output_dir) / path_obj


def main():
    parser = argparse.ArgumentParser(
        description="Create numerical multicontext audio datasets from tokenized audio text."
    )
    parser.add_argument("-t", "--train_input", required=True, help="Path to tokenized audio text file or directory.")
    parser.add_argument("-v", "--val_input", help="Optional validation file or directory.")
    parser.add_argument("--output_dir", default=".", help="Output directory for train.bin/val.bin/meta.pkl.")
    parser.add_argument("--train_output", default="train.bin", help="Training output filename.")
    parser.add_argument("--val_output", default="val.bin", help="Validation output filename.")
    parser.add_argument("-p", "--percentage_train", type=float, default=0.9,
                        help="Training split ratio when val_input is not provided.")

    parser.add_argument("--numeric_encoding", type=str,
                        choices=["uint16", "int16", "float16", "bfloat16"],
                        default="uint16", help="Numeric encoding for the dataset.")
    parser.add_argument("--numeric_min_token", type=float, default=0.0,
                        help="Minimum token value (used for vocab size and validation).")
    parser.add_argument("--numeric_max_token", type=float, default=4097.0,
                        help="Maximum token value (used for vocab size and validation).")
    parser.add_argument("--numeric_range", action="store_true",
                        help="Validate numeric tokens against min/max bounds.")
    parser.add_argument("--numeric_vocab_size", type=int, default=None,
                        help="Override vocab size for numeric tokenization.")
    parser.add_argument("-T", "--track_token_counts", action="store_true",
                        help="Track how often each token appears and store in meta.pkl.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    template_dir = repo_root / "data" / "template"
    sys.path.insert(0, str(template_dir))
    import prepare as template_prepare
    from tokenizers import NumericTokenizer

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = _read_input_data(args.train_input)
    if args.val_input:
        val_data = _read_input_data(args.val_input)
    else:
        split_idx = int(len(train_data) * args.percentage_train)
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        if args.percentage_train == 1.0:
            val_data = None

    tokenizer = NumericTokenizer(args)

    train_output_path = _resolve_output_path(output_dir, args.train_output)
    val_output_path = _resolve_output_path(output_dir, args.val_output)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        train_ids = tokenizer.tokenize(train_data)
        val_ids = tokenizer.tokenize(val_data) if val_data else None

        dtype = template_prepare.get_numeric_numpy_dtype(args.numeric_encoding)
        template_prepare.save_tokens(train_ids, str(train_output_path), dtype)
        if val_ids is not None:
            template_prepare.save_tokens(val_ids, str(val_output_path), dtype)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
