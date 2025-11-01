#!/bin/bash
# demos/pkl_eval_demo_minipile.sh
#
# Trains and evaluates a minipile model that uses the PKL-number parameterization
# with the default sqrt(2) scaling for every component. The script mirrors the
# workflow in fake_ptq_uniform_eval_demo_minipile.sh but swaps the training and
# evaluation commands to enable PKL linear layers, embeddings, and LM head.

set -euo pipefail

EVAL_DATASET_DIR="data/minipile"
OUT_DIR="out_pkl_minipile"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256
PKL_SCALE="1.4142135623730951"

mkdir -p "$EVAL_DATASET_DIR"

echo "=== Step 1: Prepare the minipile dataset ==="
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

echo "=== Step 2: Train a PKL-parameterized reference model on minipile (if needed) ==="
if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset minipile \
    --out_dir "$OUT_DIR" \
    --n_layer 6 \
    --n_head 6 \
    --n_embd 384 \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --use_qk_norm \
    --use_qk_norm_scale \
    --use_peri_ln \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_iters 10000 \
    --eval_interval 10000 \
    --eval_iters "$EVAL_ITERS" \
    --eta_variant "iteration" \
    --compile \
    --linear_variant_attn "pkl_linear" \
    --linear_variant_mlp "pkl_linear" \
    --use_pkl_wte \
    --use_pkl_lm_head \
    --pkl_linear_scale "$PKL_SCALE" \
    --pkl_wte_scale "$PKL_SCALE" \
    --pkl_lm_head_scale "$PKL_SCALE"
else
  echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate the PKL checkpoint ==="
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --eval_only \
  --eval_dataset minipile \
  --eval_iters "$EVAL_ITERS"

echo "PKL evaluation complete. Results live in $OUT_DIR."
