#!/usr/bin/env bash
set -euo pipefail

# Demo: MeZO forward-only fine-tuning from scratch.
# Assumes a dataset already prepared in data/shakespeare_char.
python3 train_mezo.py \
  --dataset shakespeare_char \
  --out_dir out_mezo_demo \
  --max_iters 2000 \
  --batch_size 64 \
  --block_size 256 \
  --learning_rate 1e-3 \
  --mezo_epsilon 1e-3

# Demo: resume MeZO training from the checkpoint above.
python3 train_mezo.py \
  --init_from resume \
  --out_dir out_mezo_demo \
  --max_iters 4000 \
  --batch_size 64 \
  --block_size 256 \
  --learning_rate 1e-3 \
  --mezo_epsilon 1e-3
