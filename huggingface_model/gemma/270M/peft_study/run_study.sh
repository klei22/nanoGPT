#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-./gemma-peft-study}"
MODEL="${MODEL:-google/gemma-3-270m}"
DATASET="${DATASET:-flytech/python-codes-25k}"
TASKS="${TASKS:-arc_easy,hellaswag,piqa,winogrande}"
STEPS="${STEPS:-1000}"
SEEDS="${SEEDS:-42}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT"
for seed in $SEEDS; do
  far="$ROOT/attention_residual_seed_${seed}"
  lora="$ROOT/lora_seed_${seed}"
  python "$HERE/../finetune_attention_residuals.py" \
    --model_name "$MODEL" --dataset_name "$DATASET" --max_steps "$STEPS" \
    --seed "$seed" --output_dir "$far"
  python "$HERE/train_lora.py" \
    --model_name "$MODEL" --dataset_name "$DATASET" --max_steps "$STEPS" \
    --seed "$seed" --output_dir "$lora"
  python "$HERE/evaluate.py" --method attention_residual --base_model "$MODEL" \
    --checkpoint "$far" --tasks "$TASKS" --output "$far/evaluation.json"
  python "$HERE/evaluate.py" --method lora --base_model "$MODEL" \
    --checkpoint "$lora" --tasks "$TASKS" --output "$lora/evaluation.json"
done
python "$HERE/report.py" --study_dir "$ROOT"

