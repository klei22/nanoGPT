#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/outputs/low_bit_em_comparison}"

python3 "$SCRIPT_DIR/low_bit_em_comparison.py" \
  --trials "${TRIALS:-100}" \
  --angles-step "${ANGLE_STEP:-5}" \
  --output-dir "$OUTPUT_DIR"

echo "Wrote INT/TQ/all-finite E/M comparison graphs to $OUTPUT_DIR"
