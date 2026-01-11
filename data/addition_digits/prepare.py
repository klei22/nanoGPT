import argparse
import os
import pickle
import random
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny addition dataset.")
    parser.add_argument("--max-number", type=int, default=99, help="Largest addend to sample.")
    parser.add_argument("--num-samples", type=int, default=20000, help="Total addition expressions to create.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction of samples to use for training.")
    return parser.parse_args()


def generate_samples(max_number: int, num_samples: int) -> List[str]:
    rows: List[str] = []
    for _ in range(num_samples):
        a = random.randint(0, max_number)
        b = random.randint(0, max_number)
        rows.append(f"{a}+{b}={a + b}\n")
    return rows


def build_vocab(text: str) -> Tuple[dict, dict]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict) -> List[int]:
    return [stoi[c] for c in text]


def save_bin(ids: List[int], path: str) -> None:
    np.array(ids, dtype=np.uint16).tofile(path)


def main() -> None:
    args = parse_args()
    samples = generate_samples(args.max_number, args.num_samples)
    full_text = "".join(samples)

    print(f"Generated {len(samples)} expressions with max addend {args.max_number}.")

    # Save raw text for inspection
    with open("addition.txt", "w", encoding="utf-8") as f:
        f.writelines(samples)

    train_cutoff = int(len(full_text) * args.train_ratio)
    train_text, val_text = full_text[:train_cutoff], full_text[train_cutoff:]

    stoi, itos = build_vocab(full_text)
    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    print(f"Vocab size: {len(stoi)} | Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

    save_bin(train_ids, "train.bin")
    save_bin(val_ids, "val.bin")

    meta = {"vocab_size": len(stoi), "itos": itos, "stoi": stoi}
    with open("meta.pkl", "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()
