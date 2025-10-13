#!/bin/bash
# demos/shakespeare_checkpoint_comparison_demo.sh
# Demonstrates training two Shakespeare character models and comparing their checkpoints.

set -euo pipefail

SKIP_TRAINING="${1:-no}"
DATA_DIR="data/shakespeare_char"
OUT_DIR_A="out_shakespeare_checkpoint_comparison_a"
OUT_DIR_B="out_shakespeare_checkpoint_comparison_b"
CKPT_PATH_A="${OUT_DIR_A}/ckpt.pt"
CKPT_PATH_B="${OUT_DIR_B}/ckpt.pt"
HIST_DIR="comparison_demo_histograms"
COMPARISON_DIR="comparison_demo_reports"
REGEX_ATTN="transformer\\.h\\.[0-9]+\\.attn\\.(c_attn_(q|k|v)|c_proj)\\.weight"
REGEX_MLP="transformer\\.h\\.[0-9]+\\.mlp\\.(c_fc|c_proj)\\.weight"

# Ensure the dataset is available.
pushd "${DATA_DIR}" > /dev/null
bash get_dataset.sh
popd > /dev/null

if [[ "${SKIP_TRAINING}" = "no" ]]; then
  echo "\nTraining first comparison checkpoint (seed 1337)..."
  rm -rf "${OUT_DIR_A}"
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${OUT_DIR_A}" \
    --max_iters 800 \
    --attention_variant infinite \
    --n_qk_head_dim 120 \
    --n_v_head_dim 120 \
    --use_concat_heads \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --block_size 64 \
    --batch_size 64 \
    --n_layer 10 \
    --n_head 6 \
    --n_embd 384 \
    --eval_interval 100 \
    --log_interval 10 \
    --compile \
    --seed 1337

  echo "\nTraining second comparison checkpoint (seed 2024)..."
  rm -rf "${OUT_DIR_B}"
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${OUT_DIR_B}" \
    --max_iters 800 \
    --attention_variant infinite \
    --n_qk_head_dim 120 \
    --n_v_head_dim 120 \
    --use_concat_heads \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --block_size 64 \
    --batch_size 64 \
    --n_layer 10 \
    --n_head 6 \
    --n_embd 384 \
    --eval_interval 100 \
    --log_interval 10 \
    --compile \
    --seed 2024
fi

if [[ ! -f "${CKPT_PATH_A}" ]]; then
  echo "Expected checkpoint not found at ${CKPT_PATH_A}" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH_B}" ]]; then
  echo "Expected checkpoint not found at ${CKPT_PATH_B}" >&2
  exit 1
fi

echo "\nRunning checkpoint regex explorer to compare attention projection weights..."
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH_A}" \
  "${CKPT_PATH_B}" \
  "${REGEX_ATTN}" \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 40 \
  --comparison-dir "${COMPARISON_DIR}" \
  --comparison-bins 60

echo "\nRunning checkpoint regex explorer to compare MLP projection weights..."
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH_A}" \
  "${CKPT_PATH_B}" \
  "${REGEX_MLP}" \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 40 \
  --comparison-dir "${COMPARISON_DIR}" \
  --comparison-bins 60

echo "\nVector angle CSVs and histograms saved under ${COMPARISON_DIR}, with per-checkpoint histograms in ${HIST_DIR}."
