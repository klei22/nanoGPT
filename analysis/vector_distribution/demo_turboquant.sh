#!/usr/bin/env bash
set -euo pipefail

# Compare TurboQuant's Gaussian Lloyd-Max codebooks with same-size integer and
# small floating-point formats. Outputs are interactive HEALPix sphere heatmaps.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/images/turboquant_comparison}"
NSIDE="${NSIDE:-32}"
mkdir -p "$OUT_DIR"

run_format() {
  local name="$1"
  shift
  python3 "$SCRIPT_DIR/vector_distribution_analysis.py" \
    "$@" --mode exhaustive --healpix --nside "$NSIDE" \
    --out3d "$OUT_DIR/healpix_${name}_nside${NSIDE}_exhaustive.html"
}

run_format int3 --format int3
run_format turboquant3 --format turboquant3
run_format int4 --format int4
run_format turboquant4 --format turboquant4
run_format fp4_e2m1 --format fp16 --exp 2 --mant 1
run_format fp6_e3m2 --format fp16 --exp 3 --mant 2

echo "Wrote TurboQuant comparison visualizations to $OUT_DIR"
