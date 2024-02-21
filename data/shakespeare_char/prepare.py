import os
import argparse
import numpy as np
import pickle
import sentencepiece as spm
import tiktoken
def tokenize_custom_tokens_and_replace(data, tokens):
    """Tokenize data using custom tokens and replace found tokens with underscores."""
    stoi = {token: i for i, token in enumerate(tokens)}
    encoded_data = []
    remaining_data = []
    i = 0
    covered_chars = 0
    while i < len(data):
        matched = False
        for token in tokens:
            if data.startswith(token, i):
                encoded_data.append(stoi[token])
                remaining_data.append("_" * len(token))
                i += len(token)
                covered_chars += len(token)
                matched = True
                break
        if not matched:
            remaining_data.append(data[i])
            i += 1  # Move to the next character if no token matches
    coverage = covered_chars / len(data)
    remaining_text = ''.join(remaining_data)
    return encoded_data, coverage, stoi, {i: token for i, token in enumerate(tokens)}, remaining_text

def tokenize_custom_tokens(data, tokens):
    """Tokenize data using custom tokens."""
    stoi = {token: i for i, token in enumerate(tokens)}
    encoded_data = []
    i = 0
    covered_chars = 0
    while i < len(data):
        matched = False
        for token in tokens:
            if data.startswith(token, i):
                encoded_data.append(stoi[token])
                i += len(token)
                covered_chars += len(token)
                matched = True
                break
        if not matched:
            i += 1  # Skip character if no token matches
    coverage = covered_chars / len(data)
    return encoded_data, coverage, stoi, {i: token for i, token in enumerate(tokens)}

def train_sentencepiece_model(input_file, model_prefix, vocab_size):
    """Train a SentencePiece model."""

    # Other options (https://github.com/google/sentencepiece/blob/master/doc/options.md)
    # self_test_sample_size=1,
    # input_format="text",
    # shuffle_input_sentence = false
    # split_digits=False, # this often helps with arithmetic
    # allow_whitespace_only_pieces=True,
    # normalization_rule_name="nmt_nfkc_cf" lower cases as well
    num_threads = os.cpu_count()
    spm.SentencePieceTrainer.train(
        num_threads=num_threads,
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
    )

    print("SentencePiece model training complete.")


def tokenize_sentencepiece(sp_model, data):
    """Tokenize data using the SentencePiece model."""
    return sp_model.encode_as_ids(data)


def tokenize_tiktoken(enc, data):
    """Tokenize data using TikToken."""
    return enc.encode_ordinary(data)


def encode_char_level(data, chars):
    """Encode data at character level."""
    stoi = {ch: i for i, ch in enumerate(chars)}
    return [stoi[ch] for ch in data], stoi, {i: ch for i, ch in enumerate(chars)}


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text data using different methods."
    )
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument(
        "--tokens_file",
        type=str,
        default=None,
        help="Path to the file containing newline-separated tokens for tokenization",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sentencepiece", "tiktoken", "char", "custom", "replace"],
        default="sentencepiece",
        help="Tokenization method",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=500,
        help="Vocabulary size for SentencePiece model",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="train.bin",
        help="Output file for tokenized training data",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="val.bin",
        help="Output file for tokenized validation data",
    )

    args = parser.parse_args()

    # Read data
    with open(args.input_file, "r") as f:
        data = f.read()
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    if args.method == "sentencepiece":
        # Train and use SentencePiece
        spm_model_prefix = os.path.splitext(args.input_file)[0] + "_spm_model"
        train_sentencepiece_model(args.input_file, spm_model_prefix, args.vocab_size)
        sp = spm.SentencePieceProcessor()
        sp.load(f"{spm_model_prefix}.model")
        train_ids = tokenize_sentencepiece(sp, train_data)
        val_ids = tokenize_sentencepiece(sp, val_data)

        # Create stoi (string-to-index) and itos (index-to-string) mappings
        stoi = {sp.id_to_piece(id): id for id in range(sp.GetPieceSize())}
        itos = {id: sp.id_to_piece(id) for id in range(sp.GetPieceSize())}

        # Manually add newline character to vocab
        if "\n" not in stoi:
            stoi["\n"] = sp.PieceToId("\n")

        # Save metadata including stoi and itos in a pickle file
        meta = {"vocab_size": sp.GetPieceSize(), "stoi": stoi, "itos": itos}
        with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    elif args.method == "tiktoken":
        # Use TikToken
        enc = tiktoken.get_encoding("gpt2")
        train_ids = tokenize_tiktoken(enc, train_data)
        val_ids = tokenize_tiktoken(enc, val_data)

    if args.method == "replace":
        if args.tokens_file is None:
            raise ValueError("Tokens file must be provided for custom tokenization method.")
        with open(args.tokens_file, "r") as f:
            tokens = [line.strip() for line in f.readlines() if line.strip()]
        train_ids, train_coverage, stoi, itos, remaining_train = tokenize_custom_tokens_and_replace(train_data, tokens)
        val_ids, val_coverage, _, _, remaining_val = tokenize_custom_tokens_and_replace(val_data, tokens)
        print(f"Training data coverage by tokens: {train_coverage*100:.2f}%")
        print(f"Validation data coverage by tokens: {val_coverage*100:.2f}%")

        # Write the remaining data (with tokens replaced by underscores) to remaining.txt
        with open("remaining.txt", "w") as f:
            f.write(remaining_train + "\n" + remaining_val)

        # Save metadata including stoi and itos in a pickle file
        meta = {"vocab_size": len(tokens), "stoi": stoi, "itos": itos}
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    elif args.method == "custom":
        if args.tokens_file is None:
            raise ValueError("Tokens file must be provided for custom tokenization method.")
        with open(args.tokens_file, "r") as f:
            tokens = [line.strip() for line in f.readlines() if line.strip()]
        train_ids, train_coverage, stoi, itos = tokenize_custom_tokens(train_data, tokens)
        val_ids, val_coverage, _, _ = tokenize_custom_tokens(val_data, tokens)
        print(f"Training data coverage by tokens: {train_coverage*100:.2f}%")
        print(f"Validation data coverage by tokens: {val_coverage*100:.2f}%")

        # Save metadata including stoi and itos in a pickle file
        meta = {"vocab_size": len(tokens), "stoi": stoi, "itos": itos}
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    # Rest of the main function remains unchanged, includ
    elif args.method == "char":
        # Print the total length of the dataset in characters
        print(f"Length of dataset in characters: {len(data):,}")
        # Character-level tokenization
        chars = sorted(list(set(train_data)))  # Get unique characters in train data
        vocab_size = len(chars)
        print("All unique characters:", "".join(chars))
        print(f"Vocab size: {vocab_size}")

        train_ids, stoi, itos = encode_char_level(train_data, chars)
        val_ids, _, _ = encode_char_level(val_data, chars)

        # Save the meta information
        meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
        with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    # Print token counts and export to bin files
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    np.array(train_ids, dtype=np.uint16).tofile(args.train_output)
    np.array(val_ids, dtype=np.uint16).tofile(args.val_output)


if __name__ == "__main__":
    main()
