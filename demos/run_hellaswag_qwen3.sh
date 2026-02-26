#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="meta-llama/Llama-3.2-1B"
DEVICE="cuda"
DTYPE="bfloat16"
SPLIT="validation"

python3 benchmarks/evaluate_huggingface_models.py \
  --model_name "$MODEL_NAME" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --split "$SPLIT"
