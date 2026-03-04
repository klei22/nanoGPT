#!/usr/bin/env bash
# Create fp16 numerical multicontext datasets from CSV columns.
# Usage examples:
#   ./get_datasets.sh
#   ./get_datasets.sh my_data.csv
#   ./get_datasets.sh my_data.csv --output_root csv_num_mc --train_ratio 0.95 \
#       --column-transform temperature:-10:0.1 --column-transform humidity:0:1.5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_CSV="${SCRIPT_DIR}/input.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  INPUT_CSV="$1"
  shift
fi

python3 "${SCRIPT_DIR}/prepare_csv_fp16_multicontext.py" \
  --input_csv "${INPUT_CSV}" \
  "$@"
