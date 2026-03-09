#!/usr/bin/env bash
set -euo pipefail

DATASET="shakespeare_char"
MAX_ITERS=1500
EXTRAP_BLOCK_SIZE=512
BASE_OUT="out_abs_pos_demo"
SUMMARY_DIR="out_abs_pos_demo_summary"

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

  # Standard and extrapolated generation runs.
  python3 sample.py --out_dir "$out_dir" --start "To be" --max_new_tokens 128 --num_samples 1 --top_k 10 > "${out_dir}_sample_default.txt"
  python3 sample.py --out_dir "$out_dir" --start "To be" --max_new_tokens 256 --num_samples 1 --top_k 10 --block_size "$EXTRAP_BLOCK_SIZE" > "${out_dir}_sample_extrapolated.txt"

  # Standard and extrapolated eval-only runs for summary plotting.
  python3 sample.py --out_dir "$out_dir" --eval_only --eval_dataset "$DATASET" --eval_iters 100 --eval_output_dir "${out_dir}_eval_default"
  python3 sample.py --out_dir "$out_dir" --eval_only --eval_dataset "$DATASET" --eval_iters 100 --block_size "$EXTRAP_BLOCK_SIZE" --eval_output_dir "${out_dir}_eval_extrap"
}

prepare_dataset

run_variant "abs_default" --use_abs_pos_embeddings --abs_pos_embedding_variant default --no-use_rotary_embeddings
run_variant "rope" --no-use_abs_pos_embeddings --use_rotary_embeddings
run_variant "abs_cyclic" --use_abs_pos_embeddings --abs_pos_embedding_variant cyclic --abs_pos_cyclic_periods 31 47 97 --abs_pos_cyclic_random_start

mkdir -p "$SUMMARY_DIR"
python3 - "$DATASET" "$EXTRAP_BLOCK_SIZE" "$SUMMARY_DIR" <<'PY'
import json
import os
import re
import statistics
import sys

import matplotlib.pyplot as plt


dataset = sys.argv[1]
extrap_block_size = int(sys.argv[2])
summary_dir = sys.argv[3]

variants = ["abs_default", "rope", "abs_cyclic"]
base_prefix = "out_abs_pos_demo"


def mean_train_ms(path):
    vals = []
    rx = re.compile(r",\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = rx.search(line)
            if m:
                vals.append(float(m.group(1)))
    if not vals:
        return None
    return statistics.mean(vals[-20:])


def read_eval_loss(path):
    with open(path, encoding="utf-8") as fh:
        return float(json.load(fh)["val"])


speed = {}
loss_default = {}
loss_extrap = {}

for v in variants:
    out_dir = f"{base_prefix}_{v}_{dataset}"
    speed[v] = mean_train_ms(f"{out_dir}_train.log")
    loss_default[v] = read_eval_loss(os.path.join(f"{out_dir}_eval_default", "eval_loss.txt"))
    loss_extrap[v] = read_eval_loss(os.path.join(f"{out_dir}_eval_extrap", "eval_loss.txt"))

labels = variants
x = range(len(labels))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].bar(labels, [speed[k] if speed[k] is not None else 0.0 for k in labels])
axes[0].set_title("Train speed")
axes[0].set_ylabel("ms / iter")

axes[1].bar(labels, [loss_default[k] for k in labels], color="#4C72B0")
axes[1].set_title("Validation loss (default block size)")
axes[1].set_ylabel("val loss")

axes[2].bar(labels, [loss_extrap[k] for k in labels], color="#55A868")
axes[2].set_title(f"Validation loss (extrap block_size={extrap_block_size})")
axes[2].set_ylabel("val loss")

fig.tight_layout()
out_png = os.path.join(summary_dir, f"absolute_position_extrapolation_{dataset}.png")
fig.savefig(out_png, dpi=150)

summary = {
    "dataset": dataset,
    "extrap_block_size": extrap_block_size,
    "speed_ms_per_iter": speed,
    "val_loss_default": loss_default,
    "val_loss_extrapolated": loss_extrap,
    "summary_graph": out_png,
}
with open(os.path.join(summary_dir, f"absolute_position_extrapolation_{dataset}.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

print(f"Wrote summary graph: {out_png}")
PY

echo "Done. Outputs in ${BASE_OUT}_*_${DATASET}. Summary graph in ${SUMMARY_DIR}."
