#!/bin/bash
# grokking_demo.sh
#
# Demonstrates the "grokking" phenomenon (Power et al., 2022) using modular
# arithmetic. The model will:
#   1. Quickly memorize the training set (train loss → 0)
#   2. Show no generalization for a long period (val loss stays high)
#   3. Eventually "grok" and suddenly generalize (val loss drops sharply)
#
# Key ingredients for grokking:
#   - Exact 50/50 train/val split of all (a+b) mod 97 examples
#   - Weight decay (1.0): strong regularization drives generalization
#   - No learning rate decay: constant LR keeps optimization pressure
#   - Long training (50k iters) to allow late generalization
#   - Small model: 2 layers, 4 heads, 128 embedding dim

set -euo pipefail

DATA_DIR="data/grokking"
OUT_DIR="out/grokking_demo"

echo "=== Step 1: Generate modular arithmetic dataset ==="
pushd "${DATA_DIR}" > /dev/null
python3 generate_dataset.py \
  --prime 97 \
  --operation addition \
  --train_fraction 0.5 \
  --seed 42

echo "=== Step 2: Tokenize (using custom prepare.py for exact train/val split) ==="
python3 prepare.py

popd > /dev/null

mkdir -p "${OUT_DIR}"

echo "=== Step 3: Train (watch for grokking: train loss drops early, val loss drops much later) ==="
python3 train.py \
  --dataset grokking \
  --out_dir "${OUT_DIR}" \
  --block_size 32 \
  --batch_size 512 \
  --n_layer 2 \
  --n_head 4 \
  --n_embd 128 \
  --max_iters 50000 \
  --eval_interval 500 \
  --eval_iters 100 \
  --log_interval 100 \
  --learning_rate 1e-3 \
  --no-decay_lr \
  --warmup_iters 10 \
  --weight_decay 1.0 \
  --dropout 0.0 \
  --csv_log \
  --always_save_checkpoint

echo ""
echo "=== Training complete ==="
echo "Output directory: ${OUT_DIR}"
echo ""
echo "Look for the grokking signature in the logs:"
echo "  - Train loss should drop to ~0 within the first few thousand iterations"
echo "  - Val loss should remain high for a long period"
echo "  - Val loss should suddenly drop much later in training"
echo ""
echo "CSV logs are available in: ${OUT_DIR}/csv_logs/"
