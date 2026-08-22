#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-out/smollm2-135m-final-attention-residual}

python -m huggingface_model.attention_residual_finetune.train \
  --output-dir "$OUTPUT_DIR" \
  "$@"

python -m huggingface_model.attention_residual_finetune.benchmark \
  --adapter "$OUTPUT_DIR/final_attention_residual.pt" \
  --output-dir "$OUTPUT_DIR/benchmarks"
