"""
Prepare a counting (histogram) dataset for nanoGPT.

Inspired by "Counting in Small Transformers" (Behrens et al., 2025).
Given a sequence of tokens, the model must predict the count of each token
in the sequence.

Format (char-level):
  Input:  "A B D D B B|"
  Target: "1 3 2 2 3 3"

The pipe '|' separates the input sequence from the expected output.
The model is trained on the full string so it learns to produce counts
after seeing the delimiter.

Usage:
  cd data/counting
  python prepare.py --T 10 --L 8 --num_train 50000 --num_val 5000
"""

import argparse
import os
import pickle
import random

import numpy as np


def generate_sequence(T, L):
    """Generate a random sequence of length L from alphabet of size T.

    Uses the partitioned sampling strategy from Behrens et al. to get
    a roughly uniform distribution over count values.
    """
    alphabet = [chr(ord('A') + i) if i < 26 else chr(ord('a') + i - 26) for i in range(T)]
    seq = [None] * L
    remaining = list(range(L))
    available_tokens = list(alphabet)

    while remaining:
        k = random.randint(1, len(remaining))
        chosen_indices = remaining[:k]
        remaining = remaining[k:]
        t = random.choice(available_tokens)
        for idx in chosen_indices:
            seq[idx] = t
        if len(available_tokens) > 1:
            available_tokens.remove(t)

    return seq


def sequence_to_example(seq):
    """Convert a sequence to input|target string.

    Example: ['A','B','D','D','B','B'] -> 'A B D D B B|1 3 2 2 3 3\n'
    """
    from collections import Counter
    counts = Counter(seq)
    input_str = ' '.join(seq)
    target_str = ' '.join(str(counts[t]) for t in seq)
    return f"{input_str}|{target_str}\n"


def build_dataset(T, L, num_samples):
    """Generate num_samples counting examples."""
    examples = []
    for _ in range(num_samples):
        seq = generate_sequence(T, L)
        examples.append(sequence_to_example(seq))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare counting/histogram dataset")
    parser.add_argument("--T", type=int, default=10,
                        help="Alphabet size (number of distinct tokens)")
    parser.add_argument("--L", type=int, default=8,
                        help="Sequence length")
    parser.add_argument("--num_train", type=int, default=50000,
                        help="Number of training examples")
    parser.add_argument("--num_val", type=int, default=5000,
                        help="Number of validation examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Generating counting dataset: T={args.T}, L={args.L}")
    print(f"  train: {args.num_train} examples, val: {args.num_val} examples")

    train_examples = build_dataset(args.T, args.L, args.num_train)
    val_examples = build_dataset(args.T, args.L, args.num_val)

    # Build character-level vocabulary from all examples
    all_text = ''.join(train_examples) + ''.join(val_examples)
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"  vocab_size: {vocab_size}")
    print(f"  chars: {''.join(chars)}")

    # Encode
    train_ids = [stoi[ch] for ex in train_examples for ch in ex]
    val_ids = [stoi[ch] for ex in val_examples for ch in ex]

    print(f"  train tokens: {len(train_ids)}, val tokens: {len(val_ids)}")

    # Save
    np.array(train_ids, dtype=np.uint16).tofile('train.bin')
    np.array(val_ids, dtype=np.uint16).tofile('val.bin')

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'tokenizer': 'char',
        'task': 'counting',
        'T': args.T,
        'L': args.L,
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    # Also save raw examples for benchmark grading
    with open('val_examples.txt', 'w') as f:
        f.writelines(val_examples)

    # Save a benchmark config
    benchmark = {
        'name': 'counting',
        'description': 'Histogram counting task: predict token frequencies in a sequence',
        'val_examples_file': 'val_examples.txt',
        'grader': 'benchmarks/grade_counting.py',
        'grader_args': f'--T {args.T} --L {args.L}',
        'max_new_tokens': args.L * 2 + args.L - 1,
        'temperature': 0.0,
        'top_k': 1,
        'num_eval_examples': 100,
        'T': args.T,
        'L': args.L,
    }
    import json
    with open('benchmark.json', 'w') as f:
        json.dump(benchmark, f, indent=2)

    print("Done! Files written: train.bin, val.bin, meta.pkl, val_examples.txt, benchmark.json")


if __name__ == '__main__':
    main()
