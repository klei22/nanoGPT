#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/outputs/high_dimensional}"
DIMENSIONS=(1024 2048 4096 8192)
mkdir -p "$OUTPUT_DIR"

python3 "$SCRIPT_DIR/high_dim_angle_space_evenness.py" \
  --dimensions "${DIMENSIONS[@]}" \
  --samples "${EVENNESS_SAMPLES:-10000}" \
  --batch-size "${EVENNESS_BATCH_SIZE:-256}" \
  --output-dir "$OUTPUT_DIR/evenness"

for dimension in "${DIMENSIONS[@]}"; do
  python3 "$SCRIPT_DIR/turboquant_angular_distortion.py" \
    --dim "$dimension" --trials "${DISTORTION_TRIALS:-100}" \
    --angles-step "${ANGLE_STEP:-5}" --pair-mode isotropic --no-transformed-tq \
    --output "$OUTPUT_DIR/isotropic_distortion_d${dimension}.pdf"
done

echo "Wrote high-dimensional evenness and distortion graphs to $OUTPUT_DIR"
