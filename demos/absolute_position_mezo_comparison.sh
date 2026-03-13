#!/usr/bin/env bash
set -euo pipefail

DATASET="shakespeare_char"
MAX_ITERS=800
BASE_OUT="out_abs_pos_mezo"
SUMMARY_DIR="out_abs_pos_mezo_summary"

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

mkdir -p "$SUMMARY_DIR"
python3 - "$DATASET" "$SUMMARY_DIR" <<'PY'
import json
import os
import sys

import matplotlib.pyplot as plt


dataset = sys.argv[1]
summary_dir = sys.argv[2]
variants = ["abs_default", "rope", "abs_cyclic"]
base_out = "out_abs_pos_mezo"

losses = {}
for v in variants:
    path = f"{base_out}_{v}_{dataset}_eval/eval_loss.txt"
    with open(path, encoding="utf-8") as fh:
        losses[v] = float(json.load(fh)["val"])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(variants, [losses[v] for v in variants])
ax.set_title(f"MeZO final validation loss ({dataset})")
ax.set_ylabel("val loss")
fig.tight_layout()

out_png = os.path.join(summary_dir, f"absolute_position_mezo_{dataset}.png")
fig.savefig(out_png, dpi=150)

with open(os.path.join(summary_dir, f"absolute_position_mezo_{dataset}.json"), "w", encoding="utf-8") as fh:
    json.dump({"dataset": dataset, "val_loss": losses, "summary_graph": out_png}, fh, indent=2)

print(f"Wrote summary graph: {out_png}")
PY

echo "Completed MeZO comparison. Summary graph in ${SUMMARY_DIR}."
