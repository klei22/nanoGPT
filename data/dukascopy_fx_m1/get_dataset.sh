#!/usr/bin/env bash
# Convert downloaded Dukascopy M1 CSV(.gz) candles into regular integer
# multicontext datasets by reusing data/csv_mc_int's pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RAW_INPUT="${1:-${SCRIPT_DIR}/raw/eurusd}"
INTEGER_CSV="${DUKASCOPY_MC_INPUT_CSV:-${SCRIPT_DIR}/input.csv}"
OUTPUT_ROOT="${DUKASCOPY_MC_OUTPUT_ROOT:-dukascopy_fx_m1}"
PRICE_MIN="${DUKASCOPY_PRICE_MIN:-0}"
PRICE_MAX="${DUKASCOPY_PRICE_MAX:-300000}"
VOLUME_MAX="${DUKASCOPY_VOLUME_MAX:-100000000}"

python3 "${SCRIPT_DIR}/prepare_dukascopy_csv_for_mc_int.py" \
  "$RAW_INPUT" \
  --output_csv "$INTEGER_CSV" \
  --price-scale "${DUKASCOPY_PRICE_SCALE:-100000}" \
  --volume-scale "${DUKASCOPY_VOLUME_SCALE:-1000}" \
  ${DUKASCOPY_INCLUDE_WEEKDAY:+--include-weekday}

CSV_MC_ARGS=(
  "$INTEGER_CSV"
  --output_root "$OUTPUT_ROOT"
  --train_ratio "${DUKASCOPY_TRAIN_RATIO:-0.9}"
  --range minute_of_day:0:1439
  --range open_ticks:"$PRICE_MIN":"$PRICE_MAX"
  --range high_ticks:"$PRICE_MIN":"$PRICE_MAX"
  --range low_ticks:"$PRICE_MIN":"$PRICE_MAX"
  --range close_ticks:"$PRICE_MIN":"$PRICE_MAX"
  --range volume_ticks:0:"$VOLUME_MAX"
)

if [[ -n "${DUKASCOPY_INCLUDE_WEEKDAY:-}" ]]; then
  CSV_MC_ARGS+=(--range weekday:0:6)
fi

"${REPO_ROOT}/data/csv_mc_int/get_dataset.sh" "${CSV_MC_ARGS[@]}"
