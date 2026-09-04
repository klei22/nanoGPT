#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_accelerate
require_single_a100_process

DATASET="${DATASET:-roneneldan/TinyStories}"
TOKENIZER="${TOKENIZER:-gpt2}"
DATASET_REVISION="${DATASET_REVISION:-}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-}"
MAX_STEPS="${MAX_STEPS:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/a100_train_$(run_stamp)_$$}"

TRAIN_ARGS=(
  --mode train
  --hardware-profile a100-80gb
  --dataset "$DATASET"
  --tokenizer "$TOKENIZER"
  --max-steps "$MAX_STEPS"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$DATASET_REVISION" ]]; then
  TRAIN_ARGS+=(--dataset-revision "$DATASET_REVISION")
fi
if [[ -n "$TOKENIZER_REVISION" ]]; then
  TRAIN_ARGS+=(--tokenizer-revision "$TOKENIZER_REVISION")
fi

exec "$PYTHON_BIN" -m accelerate.commands.launch \
  --num_machines 1 \
  --num_processes 1 \
  --mixed_precision bf16 \
  "$MESH_SCRIPT" \
  "${TRAIN_ARGS[@]}" \
  "$@"
