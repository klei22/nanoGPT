import argparse
import os
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare grokking dataset for training.")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--token_file", default=None)
    parser.add_argument("--output_dir", default=".")
    return parser.parse_args()


def load_tokens(token_file, train_text, val_text):
    if token_file:
        with open(token_file, "r", encoding="utf-8") as f:
            token_text = f.read()
    else:
        token_text = train_text + val_text
    chars = sorted(list(set(token_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def save_ids(ids, path):
    np.array(ids, dtype=np.uint16).tofile(path)


def main():
    args = parse_args()

    with open(args.train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(args.val_file, "r", encoding="utf-8") as f:
        val_text = f.read()

    stoi, itos = load_tokens(args.token_file, train_text, val_text)

    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    os.makedirs(args.output_dir, exist_ok=True)
    train_bin = os.path.join(args.output_dir, "train.bin")
    val_bin = os.path.join(args.output_dir, "val.bin")
    meta_path = os.path.join(args.output_dir, "meta.pkl")

    save_ids(train_ids, train_bin)
    save_ids(val_ids, val_bin)

    meta = {"vocab_size": len(stoi), "itos": itos, "stoi": stoi}
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    print(f"Vocab size: {len(stoi):,}")


if __name__ == "__main__":
    main()
