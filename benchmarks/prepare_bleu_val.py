#!/usr/bin/env python3
"""
Prepare BLEU validation examples from a dataset's val.bin + meta.pkl.

Takes the validation data, splits it into line-based examples, and creates
a val_examples.txt where each line has format: <prompt>|<continuation>

The prompt is the first `prompt_tokens` tokens and the continuation is
the next `target_tokens` tokens.

Usage:
    python benchmarks/prepare_bleu_val.py \
        --dataset data/opus-100 \
        --prompt-tokens 50 --target-tokens 50 \
        --num-examples 200 --output benchmark_bleu.json

This also works with any char or tiktoken encoded dataset.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

# Add repo root to path for tokenizer access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_decode_fn(meta):
    """Build decode function from meta.pkl."""
    tokenizer_type = meta.get('tokenizer', '')

    if 'itos' in meta:
        itos = meta['itos']
        return lambda ids: ''.join(itos[i] for i in ids)
    elif tokenizer_type == 'tiktoken' or meta.get('vocab_size', 0) > 1000:
        try:
            import tiktoken
            enc = tiktoken.get_encoding(meta.get('tiktoken_encoding', 'gpt2'))
            return lambda ids: enc.decode(ids)
        except ImportError:
            pass

    # Fallback: return ids as space-separated string
    return lambda ids: ' '.join(str(i) for i in ids)


def main():
    parser = argparse.ArgumentParser(description="Prepare BLEU validation data")
    parser.add_argument('--dataset', required=True,
                        help="Path to dataset directory (containing val.bin and meta.pkl)")
    parser.add_argument('--prompt-tokens', type=int, default=50,
                        help="Number of tokens for the prompt")
    parser.add_argument('--target-tokens', type=int, default=50,
                        help="Number of tokens for the target continuation")
    parser.add_argument('--num-examples', type=int, default=200,
                        help="Number of examples to extract")
    parser.add_argument('--output', type=str, default=None,
                        help="Output benchmark JSON path (default: <dataset>/benchmark_bleu.json)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset, 'meta.pkl')
    val_path = os.path.join(args.dataset, 'val.bin')

    if not os.path.exists(meta_path):
        sys.exit(f"meta.pkl not found at {meta_path}")
    if not os.path.exists(val_path):
        sys.exit(f"val.bin not found at {val_path}")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    vocab_size = meta.get('vocab_size', 0)
    dtype = np.uint32 if vocab_size > 65535 else np.uint16
    val_data = np.memmap(val_path, dtype=dtype, mode='r')

    decode = get_decode_fn(meta)

    rng = np.random.RandomState(args.seed)
    chunk_size = args.prompt_tokens + args.target_tokens

    if len(val_data) < chunk_size:
        sys.exit(f"val.bin too small: {len(val_data)} tokens, need at least {chunk_size}")

    max_start = len(val_data) - chunk_size
    starts = rng.choice(max_start, size=min(args.num_examples, max_start), replace=False)
    starts.sort()

    val_examples_path = os.path.join(args.dataset, 'val_bleu_examples.txt')
    examples = []
    with open(val_examples_path, 'w') as f:
        for start in starts:
            chunk = val_data[start:start + chunk_size].astype(int).tolist()
            prompt_text = decode(chunk[:args.prompt_tokens])
            target_text = decode(chunk[args.prompt_tokens:])
            # Use | as delimiter (escape any | in text)
            prompt_clean = prompt_text.replace('|', '/')
            target_clean = target_text.replace('|', '/')
            f.write(f"{prompt_clean}|{target_clean}\n")
            examples.append((prompt_clean, target_clean))

    output_path = args.output or os.path.join(args.dataset, 'benchmark_bleu.json')
    benchmark = {
        'name': 'bleu',
        'description': f'BLEU score on validation data from {os.path.basename(args.dataset)}',
        'val_examples_file': os.path.basename(val_examples_path),
        'grader': 'benchmarks/grade_bleu.py',
        'grader_args': '',
        'max_new_tokens': args.target_tokens,
        'temperature': 0.8,
        'top_k': 200,
        'num_eval_examples': args.num_examples,
    }
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2)

    print(f"Created {len(examples)} BLEU validation examples")
    print(f"  val_examples: {val_examples_path}")
    print(f"  benchmark config: {output_path}")


if __name__ == '__main__':
    main()
