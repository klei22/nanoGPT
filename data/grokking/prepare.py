#!/usr/bin/env python3
"""
Custom prepare.py for grokking experiments.

Unlike the template prepare.py which does a 90/10 sequential split of a single
file, this script reads the separate train_raw.txt and val_raw.txt files
produced by generate_dataset.py to preserve the exact 50/50 split of modular
arithmetic examples. This is critical for grokking — the model must be trained
on only half the examples so it needs to learn the underlying algorithm to
generalize to the held-out half.
"""

import os
import pickle
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="train_raw.txt")
    parser.add_argument("--val_file", default="val_raw.txt")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    with open(args.train_file, "r") as f:
        train_data = f.read()
    with open(args.val_file, "r") as f:
        val_data = f.read()

    # Build vocabulary from both splits
    all_chars = sorted(list(set(train_data + val_data)))
    vocab_size = len(all_chars)

    stoi = {ch: i for i, ch in enumerate(all_chars)}
    itos = {i: ch for i, ch in enumerate(all_chars)}

    print(f"Vocab size: {vocab_size}")
    print(f"Characters: {''.join(all_chars)!r}")
    print(f"Train chars: {len(train_data):,}")
    print(f"Val chars:   {len(val_data):,}")

    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")

    train_ids.tofile(os.path.join(args.output_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.output_dir, "val.bin"))

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "tokenizer": "char",
    }
    with open(os.path.join(args.output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("Saved train.bin, val.bin, meta.pkl")


if __name__ == "__main__":
    main()
