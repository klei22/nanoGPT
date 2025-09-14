# prepare.py
import json
import os
import argparse
import numpy as np
from tokenizers import (
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    ByteTokenizer,
    FileByteTokenizer,
    CharTokenizer,
    CustomCharTokenizerWithByteFallback,
    JsonByteTokenizerWithByteFallback,
)
from tqdm import tqdm
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize text data using different methods.")

    # Input/output arguments
    parser.add_argument("-t", "--train_input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("-v", "--val_input", type=str, help="Path to validation input file. If not provided, train_input will be split using percentage_train")
    parser.add_argument("--train_output", type=str, default="train.bin", help="Path to save the training output file")
    parser.add_argument("--val_output", type=str, default="val.bin", help="Path to save the validation output file")
    parser.add_argument("-p", "--percentage_train", type=float, default=0.9, help="Percentage of data to use for training (between 0 and 1) when val_input is not provided")

    # Tokenizer selection and configuration
    parser.add_argument("--method", type=str,
                       choices=["sentencepiece", "tiktoken", "char", "custom", "byte", "file_byte", "custom_char_byte_fallback", "json_byte_fallback"],
                       default="tiktoken", help="Tokenization method")

    # SentencePiece arguments
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size for SentencePiece model")
    parser.add_argument("--spm_model_file", type=str, default=None, help="Path to the pre-trained SentencePiece model file")
    parser.add_argument("--spm_vocab_file", type=str, default=None, help="Path to the SentencePiece vocabulary file")
    parser.add_argument("--skip_tokenization", action="store_true", help="Skip creation of .bin files")

    # Tiktoken arguments
    parser.add_argument("-e", "--tiktoken_encoding",
                       choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"],
                       default="gpt2", help="Version of tiktoken encoding to utilize")
    parser.add_argument("--additional_tokens_file", type=str, default=None,
                       help="Path to JSON file containing additional special tokens for tiktoken (format: {'token': id})")

    # Char tokenizer arguments
    parser.add_argument("--reuse_chars", action="store_true", help="Reuse character list from meta.pkl")

    # Custom tokenizer arguments
    parser.add_argument("--tokens_file", type=str, default=None, help="Path to the file containing newline-separated tokens for tokenization")
    parser.add_argument("--custom_chars_file", type=str, default=None, help="Path to the file containing custom characters for the tokenizer")
    parser.add_argument("--json_tokens_file", type=str, default=None, help="Path to JSON file containing tokens for json_byte_fallback tokenizer")

    # Additional options
    parser.add_argument("-T", "--track_token_counts", action="store_true", help="Track how often each token appears and store in meta.pkl")

    return parser.parse_args()

def save_tokens(ids, output_file, dtype):
    """Save tokenized data to a binary file with progress bar."""
    total = len(ids)
    batch_size = 1024 * 1024  # 1 million tokens per batch
    with open(output_file, 'wb') as f_out:
        for i in tqdm(range(0, total, batch_size), desc=f"Saving {output_file}"):
            batch = ids[i:i+batch_size]
            np.array(batch, dtype=dtype).tofile(f_out)

def main():
    args = parse_arguments()

    # Load training data (binary mode for file_byte tokenizer)
    read_mode = 'rb' if args.method == 'file_byte' else 'r'
    with open(args.train_input, read_mode) as f:
        train_data = f.read()

    # Handle validation data based on mode
    if args.val_input:
        # Direct train/val files mode
        with open(args.val_input, read_mode) as f:
            val_data = f.read()
    else:
        # Automatic splitting mode
        n = len(train_data)
        train_data, val_data = train_data[:int(n * args.percentage_train)], train_data[int(n * args.percentage_train):]
        if args.percentage_train == 1.0:
            val_data = None

    # Initialize tokenizer based on method
    if args.method == "sentencepiece":
        tokenizer = SentencePieceTokenizer(args, input_files=args.train_input)
    elif args.method == "tiktoken":
        tokenizer = TiktokenTokenizer(args)
    elif args.method == "custom":
        tokenizer = CustomTokenizer(args)
    elif args.method == "byte":
        tokenizer = ByteTokenizer(args)
    elif args.method == "file_byte":
        tokenizer = FileByteTokenizer(args)
    elif args.method == "char":
        tokenizer = CharTokenizer(args, train_data, val_data)
    elif args.method == "custom_char_byte_fallback":
        tokenizer = CustomCharTokenizerWithByteFallback(args)
    elif args.method == "json_byte_fallback":
        tokenizer = JsonByteTokenizerWithByteFallback(args)
    else:
        raise ValueError(f"Unknown tokenization method: {args.method}")

    # Tokenize data
    train_ids = tokenizer.tokenize(train_data)
    val_ids = tokenizer.tokenize(val_data) if val_data is not None else None

    # Determine dtype based on vocabulary size from meta.pkl
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    dtype = np.uint32 if vocab_size > 65535 else np.uint16

    # Save tokenized data
    save_tokens(train_ids, args.train_output, dtype)
    if val_ids is not None:
        save_tokens(val_ids, args.val_output, dtype)

    # Save additional metadata for tiktoken if needed
    if args.method == "tiktoken" and args.additional_tokens_file:
        with open(args.additional_tokens_file, 'r') as f:
            additional_tokens = json.load(f)
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        meta.update({
            "has_additional_tokens": True,
            "special_tokens": additional_tokens,
            "tokenizer": "tiktoken",
            "tiktoken_encoding": args.tiktoken_encoding
        })
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

if __name__ == "__main__":
    main()

