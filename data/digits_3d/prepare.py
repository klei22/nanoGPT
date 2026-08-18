#!/usr/bin/env python3
"""Build a configurable character vocabulary for the 3D token demo."""

import argparse
import pickle
from pathlib import Path

import numpy as np


TRAINED_SYMBOLS = "0123456789!@#$%^&*()[]{}<>?/|+-=_~:;,."
HELD_OUT_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_dataset(
    out_dir: Path,
    train_repeats: int,
    val_repeats: int,
    num_digits: int = 10,
    num_letters: int = 10,
) -> None:
    if train_repeats < 2 or val_repeats < 2:
        raise ValueError("repeat counts must be at least 2")

    if not 1 <= num_digits <= len(TRAINED_SYMBOLS):
        raise ValueError(f"num_digits must be between 1 and {len(TRAINED_SYMBOLS)}")
    if not 0 <= num_letters <= len(HELD_OUT_SYMBOLS):
        raise ValueError(f"num_letters must be between 0 and {len(HELD_OUT_SYMBOLS)}")
    sequence = TRAINED_SYMBOLS[:num_digits]
    held_out = HELD_OUT_SYMBOLS[:num_letters]
    vocab = sequence + held_out

    out_dir.mkdir(parents=True, exist_ok=True)
    stoi = {char: index for index, char in enumerate(vocab)}
    itos = {index: char for char, index in stoi.items()}

    def write_split(name: str, repeats: int) -> None:
        # Deliberately encode only digits. The letters remain valid vocabulary
        # entries so their untrained vectors can be compared with trained ones.
        ids = np.asarray([stoi[c] for c in sequence * repeats], dtype=np.uint16)
        ids.tofile(out_dir / f"{name}.bin")

    write_split("train", train_repeats)
    write_split("val", val_repeats)
    metadata = {
        "vocab_size": len(vocab),
        "stoi": stoi,
        "itos": itos,
        "train_tokens": len(sequence) * train_repeats,
        "val_tokens": len(sequence) * val_repeats,
        "trained_tokens": list(sequence),
        "unseen_tokens": list(held_out),
        "description": f"Repeated {sequence}; {held_out or 'no symbols'} are vocabulary-only controls.",
    }
    with (out_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--train-repeats", type=int, default=2000)
    parser.add_argument("--val-repeats", type=int, default=200)
    parser.add_argument("--num-digits", type=int, default=10, help="Number of trained symbols (0-9, then punctuation).")
    parser.add_argument("--num-letters", type=int, default=10, help="Number of held-out alphabetic vocabulary symbols.")
    args = parser.parse_args()
    build_dataset(args.out_dir, args.train_repeats, args.val_repeats, args.num_digits, args.num_letters)
    print(f"Wrote {args.num_digits} trained and {args.num_letters} held-out symbols to {args.out_dir}")


if __name__ == "__main__":
    main()
