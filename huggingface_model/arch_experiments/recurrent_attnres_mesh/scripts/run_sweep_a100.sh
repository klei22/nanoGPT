#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_single_a100_process

DATASET="${DATASET:-roneneldan/TinyStories}"
TOKENIZER="${TOKENIZER:-gpt2}"
MAX_STEPS="${MAX_STEPS:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/a100_tied_grid_$(run_stamp)_$$}"
if [[ "${FULL_GRID:-0}" == "1" ]]; then
  ATTENTION_MODULES="${ATTENTION_MODULES:-0,1,2,4}"
  FFN_MODULES="${FFN_MODULES:-0,1,2,4}"
else
  ATTENTION_MODULES="${ATTENTION_MODULES:-1,2,4}"
  FFN_MODULES="${FFN_MODULES:-1,2,4}"
fi
ITERATIONS="${ITERATIONS:-1,2,4,6,8}"

exec "$PYTHON_BIN" "$MESH_SCRIPT" \
  --mode sweep \
  --hardware-profile a100-80gb \
  --dataset "$DATASET" \
  --tokenizer "$TOKENIZER" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUTPUT_DIR" \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --sweep-attention-modules "$ATTENTION_MODULES" \
  --sweep-ffn-modules "$FFN_MODULES" \
  --sweep-iterations "$ITERATIONS" \
  --sweep-router-types attnres \
  --sweep-share-iteration-weights tied \
  --no-sweep-save-models \
  --sweep-stop-on-error \
  --no-sweep-stop-on-oom \
  "$@"
