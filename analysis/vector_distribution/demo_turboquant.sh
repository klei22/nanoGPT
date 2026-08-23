#!/usr/bin/env bash
set -euo pipefail

# Run the TurboQuant visualization, angular-distortion, and angular-space
# evenness analyses as one reproducible comparison suite.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ANGULAR_DIR="$ANALYSIS_DIR/turboquant_angular_distortion"
OUT_DIR="${1:-$SCRIPT_DIR/images/turboquant_comparison}"
NSIDE="${NSIDE:-32}"
ANGULAR_DIM="${ANGULAR_DIM:-4096}"
ANGULAR_TRIALS="${ANGULAR_TRIALS:-30}"
ANGLE_STEP="${ANGLE_STEP:-3}"
EVENNESS_SAMPLES="${EVENNESS_SAMPLES:-100000}"
EVENNESS_CAPS="${EVENNESS_CAPS:-256}"
DIM_SWEEP_TRIALS="${DIM_SWEEP_TRIALS:-100}"
DIM_SWEEP_ANGLE_STEP="${DIM_SWEEP_ANGLE_STEP:-5}"
HIGH_DIM_EVENNESS_SAMPLES="${HIGH_DIM_EVENNESS_SAMPLES:-10000}"
HIGH_DIM_EVENNESS_BATCH_SIZE="${HIGH_DIM_EVENNESS_BATCH_SIZE:-256}"
HIGH_DIM_DISTORTION_TRIALS="${HIGH_DIM_DISTORTION_TRIALS:-100}"
HIGH_DIM_ANGLE_STEP="${HIGH_DIM_ANGLE_STEP:-5}"
LOW_BIT_EM_TRIALS="${LOW_BIT_EM_TRIALS:-100}"
LOW_BIT_EM_ANGLE_STEP="${LOW_BIT_EM_ANGLE_STEP:-5}"
GROUPED_QUANT_TRIALS="${GROUPED_QUANT_TRIALS:-100}"
GROUPED_QUANT_ANGLE_STEP="${GROUPED_QUANT_ANGLE_STEP:-5}"
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

echo "Angular distortion: sparse-pair transform stress test"
python3 "$ANGULAR_DIR/turboquant_angular_distortion.py" \
  --dim "$ANGULAR_DIM" --trials "$ANGULAR_TRIALS" --angles-step "$ANGLE_STEP" \
  --pair-mode sparse \
  --output "$OUT_DIR/turboquant_vs_int_angular_distortion_sparse.pdf"

echo "Angular distortion: isotropic-pair baseline"
python3 "$ANGULAR_DIR/turboquant_angular_distortion.py" \
  --dim "$ANGULAR_DIM" --trials "$ANGULAR_TRIALS" --angles-step "$ANGLE_STEP" \
  --pair-mode isotropic --no-transformed-tq \
  --output "$OUT_DIR/turboquant_vs_int_angular_distortion_isotropic.pdf"

echo "Angular-space evenness and dispersion"
python3 "$ANGULAR_DIR/angle_space_evenness.py" \
  --samples "$EVENNESS_SAMPLES" --nside "$NSIDE" --caps "$EVENNESS_CAPS" \
  --csv "$OUT_DIR/angle_space_evenness.csv" \
  --output "$OUT_DIR/angle_space_evenness.pdf"

echo "Isotropic dimension sweep: d=256 through d=2048"
python3 "$ANGULAR_DIR/isotropic_dimension_sweep.py" \
  --min-dim 256 --max-dim 2048 --trials "$DIM_SWEEP_TRIALS" \
  --angles-step "$DIM_SWEEP_ANGLE_STEP" \
  --output-dir "$OUT_DIR/isotropic_dimension_sweep"

echo "Separate high-dimensional evenness and distortion graph sets"
EVENNESS_SAMPLES="$HIGH_DIM_EVENNESS_SAMPLES" \
EVENNESS_BATCH_SIZE="$HIGH_DIM_EVENNESS_BATCH_SIZE" \
DISTORTION_TRIALS="$HIGH_DIM_DISTORTION_TRIALS" \
ANGLE_STEP="$HIGH_DIM_ANGLE_STEP" \
bash "$ANGULAR_DIR/demo_high_dimensional.sh" "$OUT_DIR/high_dimensional"

echo "INT3/INT4 vs TQ3/TQ4 vs all-finite E/M formats"
TRIALS="$LOW_BIT_EM_TRIALS" ANGLE_STEP="$LOW_BIT_EM_ANGLE_STEP" \
bash "$ANGULAR_DIR/demo_low_bit_em.sh" "$OUT_DIR/low_bit_em_comparison"

echo "Symmetric/asymmetric scalar and power-of-two grouped quantization"
TRIALS="$GROUPED_QUANT_TRIALS" ANGLE_STEP="$GROUPED_QUANT_ANGLE_STEP" \
bash "$ANGULAR_DIR/demo_grouped_quantization.sh" "$OUT_DIR/grouped_quantization_sweep"

echo "Wrote the complete TurboQuant analysis suite to $OUT_DIR"
