#!/usr/bin/env bash
# Dukascopy FX M1 regular integer CSV multicontext demo:
# 1) download a tiny default Dukascopy candle slice when raw CSVs are missing
# 2) convert downloaded Dukascopy candle CSV(.gz) files into integer columns
# 3) split those columns into per-column integer datasets via data/csv_mc_int
# 4) train regular multicontext and sample timestamped CSV continuations

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
DOWNLOAD_START="${DUKASCOPY_DEMO_START:-2025-01-02}"
DOWNLOAD_END="${DUKASCOPY_DEMO_END:-2025-01-03}"
DOWNLOAD_SIDE="${DUKASCOPY_DEMO_SIDE:-BID}"
DOWNLOAD_UNIVERSE="${DUKASCOPY_DEMO_UNIVERSE:-majors}"
DOWNLOAD_OUT="${DUKASCOPY_DEMO_RAW_OUT:-data/dukascopy_fx_m1/raw}"

has_candle_csvs() {
  local path="$1"
  if [[ -f "$path" ]]; then
    return 0
  fi
  if [[ -d "$path" ]] && find "$path" -type f \( -name "*.csv" -o -name "*.csv.gz" \) -print -quit | grep -q .; then
    return 0
  fi
  return 1
}

if ! has_candle_csvs "$RAW_INPUT"; then
  echo "No Dukascopy candle CSVs found at $RAW_INPUT; downloading $DOWNLOAD_UNIVERSE $DOWNLOAD_SIDE data for [$DOWNLOAD_START, $DOWNLOAD_END)."
  python3 data/dukascopy_fx_m1/download_dukascopy_fx_m1.py \
    --start "$DOWNLOAD_START" \
    --end "$DOWNLOAD_END" \
    --universe "$DOWNLOAD_UNIVERSE" \
    --out "$DOWNLOAD_OUT" \
    --side "$DOWNLOAD_SIDE" \
    --max-workers "${DUKASCOPY_DEMO_MAX_WORKERS:-4}" \
    --rps "${DUKASCOPY_DEMO_RPS:-2}"
fi

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

read -r -a VIEWER_SEEDS <<<"${DUKASCOPY_VIEWER_SEEDS:-1337 1338 1339}"
read -r -a VIEWER_TOP_K <<<"${DUKASCOPY_VIEWER_TOP_K:-1 5}"

python3 data/timeseries_viewer/generate_timeseries_comparison.py \
  --input_csv "$CSV_INPUT" \
  --manifest "data/$OUTPUT_ROOT/manifest.json" \
  --checkpoint_dir "$OUT_DIR" \
  --work_dir "${DUKASCOPY_VIEWER_WORK_DIR:-$OUT_DIR/timeseries_viewer}" \
  --holdout_rows "${DUKASCOPY_VIEWER_HOLDOUT_ROWS:-128}" \
  --prompt_rows "${DUKASCOPY_VIEWER_PROMPT_ROWS:-512}" \
  --seeds "${VIEWER_SEEDS[@]}" \
  --top_k "${VIEWER_TOP_K[@]}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --compile
