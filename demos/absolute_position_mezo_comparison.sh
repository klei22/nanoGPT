#!/usr/bin/env bash
set -euo pipefail

DATASET="shakespeare_char"
MAX_ITERS=800
BASE_OUT="out_abs_pos_mezo"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

run_one() {
  local name="$1"
  shift
  local out_dir="${BASE_OUT}_${name}_${DATASET}"
  python3 train_mezo.py \
    --dataset "$DATASET" \
    --out_dir "$out_dir" \
    --max_iters "$MAX_ITERS" \
    --batch_size 64 \
    --block_size 128 \
    --learning_rate 1e-3 \
    --mezo_epsilon 1e-3 \
    --eval_interval 200 \
    --eval_iters 50 \
    "$@"

  python3 sample.py --out_dir "$out_dir" --eval_only --eval_dataset "$DATASET" --eval_iters 100 --eval_output_dir "${out_dir}_eval"
}

run_one "abs_default" --use_abs_pos_embeddings --abs_pos_embedding_variant default --no-use_rotary_embeddings
run_one "rope" --no-use_abs_pos_embeddings --use_rotary_embeddings
run_one "abs_cyclic" --use_abs_pos_embeddings --abs_pos_embedding_variant cyclic --abs_pos_cyclic_periods 31 47 97 --abs_pos_cyclic_random_start

echo "Completed MeZO comparison. Check *_eval/eval_loss.txt files."
