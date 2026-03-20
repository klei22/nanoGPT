#!/usr/bin/env python3
"""
Grader for the counting/histogram benchmark task.

Reads generated and target texts (one per line) and computes:
- exact_match: fraction of examples where the full count sequence is correct
- token_accuracy: fraction of individual count tokens that are correct
- score: same as exact_match (primary metric)

Usage:
    python benchmarks/grade_counting.py \
        --generated-file gen.txt --target-file tgt.txt [--T 10 --L 8]
"""

import argparse
import json
import sys


def parse_counts(text):
    """Parse a count string like '1 3 2 2 3 3' into a list of ints."""
    parts = text.strip().split()
    counts = []
    for p in parts:
        try:
            counts.append(int(p))
        except ValueError:
            counts.append(-1)  # sentinel for unparseable
    return counts


def grade(generated_lines, target_lines):
    """Grade generated vs target counting outputs.

    Returns dict with exact_match, token_accuracy, and score.
    """
    exact_matches = 0
    total_tokens = 0
    correct_tokens = 0
    total_examples = 0

    for gen, tgt in zip(generated_lines, target_lines):
        gen = gen.strip()
        tgt = tgt.strip()

        if not tgt:
            continue

        total_examples += 1

        gen_counts = parse_counts(gen)
        tgt_counts = parse_counts(tgt)

        if gen_counts == tgt_counts:
            exact_matches += 1

        # Per-token accuracy
        for i in range(min(len(gen_counts), len(tgt_counts))):
            total_tokens += 1
            if gen_counts[i] == tgt_counts[i]:
                correct_tokens += 1
        # Count missing tokens as incorrect
        total_tokens += abs(len(gen_counts) - len(tgt_counts))

    exact_match = exact_matches / total_examples if total_examples > 0 else 0.0
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        'score': exact_match,
        'exact_match': exact_match,
        'token_accuracy': token_acc,
        'correct': exact_matches,
        'total': total_examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Grade counting benchmark")
    parser.add_argument('--generated-file', required=True, help="File with generated outputs")
    parser.add_argument('--target-file', required=True, help="File with target outputs")
    parser.add_argument('--T', type=int, default=10, help="Alphabet size (informational)")
    parser.add_argument('--L', type=int, default=8, help="Sequence length (informational)")
    args = parser.parse_args()

    with open(args.generated_file, 'r') as f:
        generated = f.readlines()
    with open(args.target_file, 'r') as f:
        targets = f.readlines()

    result = grade(generated, targets)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
