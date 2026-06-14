#!/bin/bash
# Quick demo: compare tiny Shakespeare-char runs while logging TensorBoard
# histograms and HTML animations of lm_head vocab-vector magnitudes.
set -euo pipefail

BASE_OUT_DIR="out/lm_head_vocab_hist_demo"
LOG_INTERVAL=10
MAX_SNAPSHOTS=32

COMMON_TRAIN_ARGS=(
  --dataset shakespeare_char
  --n_layer 2
  --n_head 2
  --n_embd 128
  --block_size 128
  --batch_size 32
  --max_iters 200
  --eval_interval 50
  --eval_iters 20
  --log_interval "${LOG_INTERVAL}"
  --learning_rate 1e-3
  --tensorboard_log
  --log_lm_head_vocab_hist
  --log_lm_head_vocab_hist_interval "${LOG_INTERVAL}"
  --export_lm_head_vocab_hist_html
  --lm_head_vocab_hist_max_snapshots "${MAX_SNAPSHOTS}"
)

run_experiment() {
  local name="$1"
  shift

  local out_dir="${BASE_OUT_DIR}/${name}"
  local run_name="lm_head_vocab_hist_${name}"

  echo "=== Running ${name} ==="
  python3 train.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --out_dir "${out_dir}" \
    --tensorboard_run_name "${run_name}" \
    --lm_head_vocab_hist_html_path "${out_dir}/lm_head_vocab_histogram.html" \
    "$@"
}

run_experiment "baseline"
run_experiment "capped_hyperspherenorm_lm_head" \
  --norm_variant_lm_head cappedhyperspherenorm

cat <<MSG
Done.
TensorBoard:
  tensorboard --logdir logs
  tags:
    shakespeare_char/lm_head_vocab_vector_magnitude
HTML reports:
  ${BASE_OUT_DIR}/baseline/lm_head_vocab_histogram.html
  ${BASE_OUT_DIR}/capped_hyperspherenorm_lm_head/lm_head_vocab_histogram.html
MSG
