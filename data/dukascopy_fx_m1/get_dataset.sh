#!/usr/bin/env bash
# Convert downloaded Dukascopy M1 CSV(.gz) candles into regular integer
# multicontext datasets by reusing data/csv_mc_int's pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RAW_INPUT="${1:-${SCRIPT_DIR}/raw/eurusd}"
INTEGER_CSV="${DUKASCOPY_MC_INPUT_CSV:-${SCRIPT_DIR}/input.csv}"
OUTPUT_ROOT="${DUKASCOPY_MC_OUTPUT_ROOT:-dukascopy_fx_m1}"
STATS_DIR="${DUKASCOPY_STATS_DIR:-${SCRIPT_DIR}/stats}"
DELTA_STATES="${DUKASCOPY_DELTA_STATES:-257}"

PREPARE_ARGS=(
  "$RAW_INPUT"
  --output_csv "$INTEGER_CSV"
  --stats-dir "$STATS_DIR"
  --price-scale "${DUKASCOPY_PRICE_SCALE:-100000}"
  --volume-scale "${DUKASCOPY_VOLUME_SCALE:-1000}"
  --delta-states "$DELTA_STATES"
  --lower-percentile "${DUKASCOPY_DELTA_LOWER_PERCENTILE:-5}"
  --upper-percentile "${DUKASCOPY_DELTA_UPPER_PERCENTILE:-95}"
)

if [[ -n "${DUKASCOPY_LOG_DELTA:-}" ]]; then
  PREPARE_ARGS+=(--log-delta)
fi

if [[ -n "${DUKASCOPY_DELTA_THRESHOLDS:-}" ]]; then
  read -r -a THRESHOLD_SPECS <<<"$DUKASCOPY_DELTA_THRESHOLDS"
  for spec in "${THRESHOLD_SPECS[@]}"; do
    PREPARE_ARGS+=(--delta-threshold "$spec")
  done
fi

python3 "${SCRIPT_DIR}/prepare_dukascopy_csv_for_mc_int.py" "${PREPARE_ARGS[@]}"

CSV_MC_ARGS=(
  "$INTEGER_CSV"
  --output_root "$OUTPUT_ROOT"
  --train_ratio "${DUKASCOPY_TRAIN_RATIO:-0.9}"
  --range minute_mod_10:0:9
  --range minute_of_hour:0:59
  --range minute_of_day:0:1439
  --range minute_of_week:0:10079
  --range minute_of_year:0:527039
  --range open_delta_state:0:$((DELTA_STATES - 1))
  --range high_delta_state:0:$((DELTA_STATES - 1))
  --range low_delta_state:0:$((DELTA_STATES - 1))
  --range close_delta_state:0:$((DELTA_STATES - 1))
  --range volume_delta_state:0:$((DELTA_STATES - 1))
)

"${REPO_ROOT}/data/csv_mc_int/get_dataset.sh" "${CSV_MC_ARGS[@]}"
