#!/usr/bin/env bash
# Create integer-quantized numerical multicontext datasets from CSV columns.
# Usage examples:
#   ./get_datasets.sh
#   ./get_datasets.sh my_data.csv
#   ./get_datasets.sh my_data.csv --output_root csv_num_mc_int --train_ratio 0.95 \
#       --column-transform bpm:-40:2 --column-transform spo2:0:10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_CSV="${SCRIPT_DIR}/input.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  INPUT_CSV="$1"
  shift
fi

python3 "${SCRIPT_DIR}/prepare_csv_int_multicontext.py" \
  --input_csv "${INPUT_CSV}" \
  "$@"
