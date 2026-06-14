#!/bin/bash
# Quick demo: train a tiny Shakespeare-char model while logging TensorBoard
# histograms and an HTML animation of lm_head vocab-vector magnitudes.
set -euo pipefail

OUT_DIR="out/lm_head_vocab_hist_demo"
RUN_NAME="lm_head_vocab_hist_demo"

python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${OUT_DIR}" \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 128 \
  --block_size 128 \
  --batch_size 32 \
  --max_iters 200 \
  --eval_interval 50 \
  --eval_iters 20 \
  --log_interval 10 \
  --learning_rate 1e-3 \
  --tensorboard_log \
  --tensorboard_run_name "${RUN_NAME}" \
  --log_lm_head_vocab_hist \
  --log_lm_head_vocab_hist_interval 10 \
  --export_lm_head_vocab_hist_html \
  --lm_head_vocab_hist_max_snapshots 32 \
  --lm_head_vocab_hist_html_path "${OUT_DIR}/lm_head_vocab_histogram.html"

cat <<MSG
Done.
TensorBoard:
  tensorboard --logdir logs
  tag: shakespeare_char/lm_head_vocab_vector_magnitude
HTML:
  ${OUT_DIR}/lm_head_vocab_histogram.html
MSG
