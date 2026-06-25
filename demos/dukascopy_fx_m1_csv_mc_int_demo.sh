#!/usr/bin/env bash
# Dukascopy FX M1 regular integer CSV multicontext demo:
# 1) convert downloaded Dukascopy candle CSV(.gz) files into integer columns
# 2) split those columns into per-column integer datasets via data/csv_mc_int
# 3) train regular multicontext and sample timestamped CSV continuations

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW_INPUT="${1:-data/dukascopy_fx_m1/raw/eurusd}"
OUTPUT_ROOT="${DUKASCOPY_MC_OUTPUT_ROOT:-dukascopy_fx_m1}"
CSV_INPUT="${DUKASCOPY_MC_INPUT_CSV:-data/dukascopy_fx_m1/input.csv}"
OUT_DIR="${DUKASCOPY_MC_OUT_DIR:-out/dukascopy_fx_m1}"
MAX_ITERS="${DUKASCOPY_MC_MAX_ITERS:-1000}"
DEVICE="${DUKASCOPY_MC_DEVICE:-cuda:0}"
DTYPE="${DUKASCOPY_MC_DTYPE:-bfloat16}"

# Reuse the same dataset preparation path as csv_mc_int_demo.sh, with a
# Dukascopy-specific float-to-integer conditioning step in front.
data/dukascopy_fx_m1/get_dataset.sh "$RAW_INPUT"

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
  --n_layer 6 \
  --n_head 6 \
  --attention_variant infinite \
  --use_concat_heads \
  --n_qk_head_dim 200 \
  --n_v_head_dim 112 \
  --n_embd 128 \
  --block_size 100 \
  --batch_size 2 \
  --max_iters "$MAX_ITERS" \
  --eval_interval 50 \
  --eval_iters 10 \
  --learning_rate 1e-3 \
  --dropout 0.0 \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --optimizer muon \
  --weight_decay 0.0 \
  --softmax_variant_attn relu2max \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --compile \
  --out_dir "$OUT_DIR"

python3 sample.py \
  --out_dir "$OUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --multicontext_csv_input "$CSV_INPUT" \
  --multicontext_csv_output_dir "$OUT_DIR/csv_samples" \
  --max_new_tokens 10000 \
  --top_k 5 \
  --compile \
  --num_samples 1
