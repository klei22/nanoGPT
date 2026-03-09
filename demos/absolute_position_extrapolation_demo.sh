#!/usr/bin/env bash
set -euo pipefail

DATASET="shakespeare_char"
MAX_ITERS=1500
EXTRAP_BLOCK_SIZE=512
BASE_OUT="out_abs_pos_demo"

usage() {
  cat <<'USAGE'
Usage: demos/absolute_position_extrapolation_demo.sh [--dataset shakespeare_char|minipile] [--max-iters N] [--extrap-block-size N]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --extrap-block-size) EXTRAP_BLOCK_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

prepare_dataset() {
  if [[ "$DATASET" == "shakespeare_char" ]]; then
    pushd data/shakespeare_char >/dev/null
    [[ -f train.bin && -f val.bin && -f meta.pkl ]] || { bash get_dataset.sh; python3 prepare.py -t input.txt; }
    popd >/dev/null
  elif [[ "$DATASET" == "minipile" ]]; then
    pushd data/minipile >/dev/null
    [[ -f train.bin && -f val.bin && -f meta.pkl ]] || { bash get_dataset.sh; python3 prepare.py -t input.txt --method tiktoken; }
    popd >/dev/null
  else
    echo "Unsupported dataset: $DATASET" >&2
    exit 1
  fi
}

run_variant() {
  local name="$1"
  shift
  local out_dir="${BASE_OUT}_${name}_${DATASET}"

  python3 train.py \
    --dataset "$DATASET" \
    --out_dir "$out_dir" \
    --n_layer 4 --n_head 4 --n_embd 256 \
    --block_size 128 --batch_size 64 \
    --max_iters "$MAX_ITERS" --eval_interval 500 --eval_iters 100 \
    --log_interval 20 \
    "$@" | tee "${out_dir}_train.log"

  python3 sample.py --out_dir "$out_dir" --start "To be" --max_new_tokens 128 --num_samples 1 --top_k 10 > "${out_dir}_sample_default.txt"
  python3 sample.py --out_dir "$out_dir" --start "To be" --max_new_tokens 256 --num_samples 1 --top_k 10 --block_size "$EXTRAP_BLOCK_SIZE" > "${out_dir}_sample_extrapolated.txt"
}

prepare_dataset

run_variant "abs_default" --use_abs_pos_embeddings --abs_pos_embedding_variant default --no-use_rotary_embeddings
run_variant "rope" --no-use_abs_pos_embeddings --use_rotary_embeddings
run_variant "abs_cyclic" --use_abs_pos_embeddings --abs_pos_embedding_variant cyclic --abs_pos_cyclic_periods 31 47 97 --abs_pos_cyclic_random_start

echo "Done. Outputs in ${BASE_OUT}_*_${DATASET}."
