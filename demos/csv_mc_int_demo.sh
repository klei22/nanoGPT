#!/usr/bin/env bash
# Generic integer CSV regular multicontext demo:
# 1) split a CSV into one integer-range dataset per column
# 2) train regular multicontext with a separate vocabulary per column
# 3) sample from a CSV prompt and write timestamped CSV continuations

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CSV_INPUT="${1:-data/csv_mc_int/input.csv}"
OUTPUT_ROOT="${CSV_MC_OUTPUT_ROOT:-csv_mc_int}"
OUT_DIR="${CSV_MC_OUT_DIR:-out/csv_mc_int}"
MAX_ITERS="${CSV_MC_MAX_ITERS:-200}"

# Override or extend these ranges for your own CSV. Every value is checked
# before any binary dataset files are written.
data/csv_mc_int/get_dataset.sh "$CSV_INPUT" \
  --output_root "$OUTPUT_ROOT" \
  --train_ratio 0.8 \
  --range time:0:10000 \
  --range temp:-100:200 \
  --range pressure:800:1200 \
  --save_values_csv

mapfile -t DATASETS < <(python3 - <<PY
import json
from pathlib import Path
manifest = json.loads(Path('data/$OUTPUT_ROOT/manifest.json').read_text())
for dataset in manifest['multicontext_datasets']:
    print(dataset)
PY
)

python3 train.py \
  --training_mode multicontext \
  --dataset "${DATASETS[0]}" \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 128 \
  --block_size 8 \
  --batch_size 4 \
  --max_iters "$MAX_ITERS" \
  --eval_interval 50 \
  --eval_iters 10 \
  --learning_rate 1e-3 \
  --dropout 0.0 \
  --device "${CSV_MC_DEVICE:-cpu}" \
  --dtype float32 \
  --no-compile \
  --out_dir "$OUT_DIR"

python3 sample.py \
  --out_dir "$OUT_DIR" \
  --device "${CSV_MC_DEVICE:-cpu}" \
  --dtype float32 \
  --no-compile \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --multicontext_csv_input "$CSV_INPUT" \
  --multicontext_csv_output_dir "$OUT_DIR/csv_samples" \
  --max_new_tokens 8 \
  --top_k 1 \
  --num_samples 3
