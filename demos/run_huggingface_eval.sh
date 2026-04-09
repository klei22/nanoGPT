#!/usr/bin/env bash
set -euo pipefail

MODEL_NAMES=(
  # "Qwen/Qwen2-1.5B"
  # "Qwen/Qwen3-0.6B"
  # "Qwen/Qwen3-1.7B"
  # "Qwen/Qwen3.5-0.8B-Base"
  # "microsoft/phi-1_5"
  # "EleutherAI/pythia-1b"
  "meta-llama/Llama-3.2-1B"
  "HuggingFaceTB/SmolLM2-1.7B"
  # "HuggingFaceTB/SmolLM2-135M"
  "google/gemma-2-2b"
  # "internlm/internlm2_5-1_8b"


)
BENCHMARKS=(
  "hellaswag"
  "arc-easy"
  "arc-challenge"
  "sciq"
  # "piqa"
  "winogrande"
  "boolq"
)
DEVICE="cuda"
DTYPE="float32"
SPLIT="validation"

BENCHMARK_LIST="$(IFS=,; echo "${BENCHMARKS[*]}")"

for model_name in "${MODEL_NAMES[@]}"; do
  echo "== Evaluating: ${model_name} on ${BENCHMARK_LIST} =="
  python3 benchmarks/evaluate_huggingface_models.py \
    --model_name "$model_name" \
    --benchmarks "$BENCHMARK_LIST" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --split "$SPLIT"
done
