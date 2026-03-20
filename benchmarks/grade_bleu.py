#!/usr/bin/env python3
"""
Grader for BLEU score benchmark.

Computes BLEU score between generated and target texts using sacrebleu
(if available) or a simple n-gram implementation as fallback.

Usage:
    python benchmarks/grade_bleu.py \
        --generated-file gen.txt --target-file tgt.txt [--smooth-method exp]
"""

import argparse
import json
import math
import sys
from collections import Counter


def _ngrams(tokens, n):
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _simple_bleu(generated_lines, target_lines, max_n=4):
    """Compute a simple corpus-level BLEU score without sacrebleu.

    Uses brevity penalty and modified n-gram precision up to max_n.
    """
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    gen_length = 0
    ref_length = 0

    for gen, ref in zip(generated_lines, target_lines):
        gen_tokens = gen.strip().split()
        ref_tokens = ref.strip().split()

        gen_length += len(gen_tokens)
        ref_length += len(ref_tokens)

        for n in range(1, max_n + 1):
            gen_ngrams = _ngrams(gen_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)

            gen_counts = Counter(gen_ngrams)
            ref_counts = Counter(ref_ngrams)

            for ngram, count in gen_counts.items():
                clipped_counts[n-1] += min(count, ref_counts.get(ngram, 0))
            total_counts[n-1] += len(gen_ngrams)

    # Compute modified precision for each n
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    # Brevity penalty
    if gen_length == 0:
        return 0.0
    bp = min(1.0, math.exp(1 - ref_length / gen_length))

    # Geometric mean of precisions (with smoothing)
    log_avg = 0.0
    for p in precisions:
        if p == 0:
            return 0.0
        log_avg += math.log(p) / max_n

    return bp * math.exp(log_avg) * 100.0  # scale to 0-100


def grade(generated_lines, target_lines, smooth_method='exp'):
    """Grade generated vs target using BLEU score.

    Returns dict with score (BLEU), plus per-n-gram precisions if available.
    """
    # Filter empty lines
    pairs = [(g.strip(), t.strip()) for g, t in zip(generated_lines, target_lines)
             if g.strip() and t.strip()]
    if not pairs:
        return {'score': 0.0, 'bleu': 0.0, 'num_pairs': 0}

    gen_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]

    try:
        import sacrebleu
        bleu_result = sacrebleu.corpus_bleu(gen_texts, [tgt_texts],
                                             smooth_method=smooth_method)
        return {
            'score': bleu_result.score,
            'bleu': bleu_result.score,
            'bp': bleu_result.bp,
            'num_pairs': len(pairs),
        }
    except ImportError:
        bleu = _simple_bleu(gen_texts, tgt_texts)
        return {
            'score': bleu,
            'bleu': bleu,
            'num_pairs': len(pairs),
            'note': 'sacrebleu not installed, using simple BLEU',
        }


def main():
    parser = argparse.ArgumentParser(description="Grade BLEU benchmark")
    parser.add_argument('--generated-file', required=True)
    parser.add_argument('--target-file', required=True)
    parser.add_argument('--smooth-method', default='exp',
                        help="Smoothing method for sacrebleu")
    args = parser.parse_args()

    with open(args.generated_file, 'r') as f:
        generated = f.readlines()
    with open(args.target_file, 'r') as f:
        targets = f.readlines()

    result = grade(generated, targets, args.smooth_method)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
