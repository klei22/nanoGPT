#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/outputs/isotropic_dimension_sweep}"

python3 "$SCRIPT_DIR/isotropic_dimension_sweep.py" \
  --min-dim "${MIN_DIM:-2}" \
  --max-dim "${MAX_DIM:-1024}" \
  --trials "${TRIALS:-100}" \
  --angles-step "${ANGLE_STEP:-5}" \
  --output-dir "$OUTPUT_DIR"

echo "Wrote isotropic dimension sweep to $OUTPUT_DIR"
