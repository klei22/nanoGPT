#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/outputs/grouped_quantization_sweep}"

python3 "$SCRIPT_DIR/grouped_quantization_sweep.py" \
  --dimension 2048 --trials "${TRIALS:-100}" \
  --angles-step "${ANGLE_STEP:-5}" --output-dir "$OUTPUT_DIR"

echo "Wrote grouped quantization comparison graphs to $OUTPUT_DIR"
