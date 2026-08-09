import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Generate grokking modular addition data.")
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--train_fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_dir", type=str, default=".")
    return parser.parse_args()


def format_example(a, b, modulus):
    c = (a + b) % modulus
    return f"{a} + {b} = {c}\n"


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    examples = [
        format_example(a, b, args.modulus)
        for a in range(args.modulus)
        for b in range(args.modulus)
    ]
    rng.shuffle(examples)

    total = len(examples)
    train_count = int(total * args.train_fraction)
    train_examples = examples[:train_count]
    val_examples = examples[train_count:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_examples)
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_examples)

    print(f"Generated {total:,} examples for modulus {args.modulus}.")
    print(f"Train split: {len(train_examples):,} -> {train_path}")
    print(f"Val split: {len(val_examples):,} -> {val_path}")


if __name__ == "__main__":
    main()
