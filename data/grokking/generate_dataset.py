#!/usr/bin/env python3
"""
Generate modular arithmetic datasets for studying the grokking phenomenon.

Grokking (Power et al., 2022) is the phenomenon where a neural network first
memorizes training data, then much later suddenly generalizes to the validation
set. This script generates the kind of small modular arithmetic datasets that
reliably produce grokking behavior.

Supported operations:
  - addition:    (a + b) mod p
  - subtraction: (a - b) mod p
  - division:    (a / b) mod p  (i.e., a * b^{-1} mod p, b != 0)
  - x2y:         (a^2 + a*b) mod p (polynomial)

The dataset is written as lines of the form:
  <a><sep><op><sep><b><sep>=<sep><c>\n

For example with p=97, addition, separator=' ':
  23 + 45 = 68
  90 + 12 = 5

Usage:
  python generate_dataset.py --prime 97 --operation addition --train_fraction 0.5 --seed 42
"""

import argparse
import random
import os


def mod_inverse(b, p):
    """Compute modular inverse of b mod p using Fermat's little theorem."""
    return pow(b, p - 2, p)


def generate_examples(prime, operation):
    """Generate all (a, b, result) triples for the given operation mod prime."""
    examples = []

    if operation == "addition":
        for a in range(prime):
            for b in range(prime):
                c = (a + b) % prime
                examples.append((a, b, c, "+"))
    elif operation == "subtraction":
        for a in range(prime):
            for b in range(prime):
                c = (a - b) % prime
                examples.append((a, b, c, "-"))
    elif operation == "division":
        for a in range(prime):
            for b in range(1, prime):  # skip b=0
                c = (a * mod_inverse(b, prime)) % prime
                examples.append((a, b, c, "/"))
    elif operation == "x2y":
        for a in range(prime):
            for b in range(prime):
                c = (a * a + a * b) % prime
                examples.append((a, b, c, "x"))
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return examples


def format_example(a, b, c, op_symbol, separator=" "):
    """Format a single example as a string."""
    return f"{a}{separator}{op_symbol}{separator}{b}{separator}={separator}{c}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate modular arithmetic dataset for grokking experiments"
    )
    parser.add_argument(
        "--prime", type=int, default=97,
        help="Prime modulus (default: 97)"
    )
    parser.add_argument(
        "--operation", type=str, default="addition",
        choices=["addition", "subtraction", "division", "x2y"],
        help="Arithmetic operation (default: addition)"
    )
    parser.add_argument(
        "--train_fraction", type=float, default=0.5,
        help="Fraction of examples for training (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--separator", type=str, default=" ",
        help="Separator between tokens (default: space)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Output directory (default: current directory)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    examples = generate_examples(args.prime, args.operation)
    random.shuffle(examples)

    split_idx = int(len(examples) * args.train_fraction)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"Operation: {args.operation} mod {args.prime}")
    print(f"Total examples: {len(examples)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Write train and val as separate files
    train_path = os.path.join(args.output_dir, "train_raw.txt")
    val_path = os.path.join(args.output_dir, "val_raw.txt")

    with open(train_path, "w") as f:
        for a, b, c, op in train_examples:
            f.write(format_example(a, b, c, op, args.separator) + "\n")

    with open(val_path, "w") as f:
        for a, b, c, op in val_examples:
            f.write(format_example(a, b, c, op, args.separator) + "\n")

    # Also write a combined input.txt (train then val) for the template prepare.py
    input_path = os.path.join(args.output_dir, "input.txt")
    with open(input_path, "w") as f:
        for a, b, c, op in train_examples:
            f.write(format_example(a, b, c, op, args.separator) + "\n")
        for a, b, c, op in val_examples:
            f.write(format_example(a, b, c, op, args.separator) + "\n")

    print(f"Written: {train_path}, {val_path}, {input_path}")


if __name__ == "__main__":
    main()
