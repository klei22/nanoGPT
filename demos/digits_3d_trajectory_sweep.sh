#!/usr/bin/env bash
# Run every embedding-radius mode across several trained/held-out vocab sizes.
set -euo pipefail

DIGIT_COUNTS="${DIGIT_COUNTS:-10}"
LETTER_COUNTS="${LETTER_COUNTS:-10}"
EMBEDDING_DIMS="${EMBEDDING_DIMS:-3}"
WTE_TYING_MODES="${WTE_TYING_MODES:-tied untied}"
SWEEP_MAX_ITERS="${SWEEP_MAX_ITERS:-10000}"
SWEEP_SAVE_INTERVAL="${SWEEP_SAVE_INTERVAL:-100}"
RUNS_DIR="report/threejs/digits-3d/runs"

mkdir -p "${RUNS_DIR}"

for embedding_dim in ${EMBEDDING_DIMS}; do
  for num_digits in ${DIGIT_COUNTS}; do
    for num_letters in ${LETTER_COUNTS}; do
      for mode in unconstrained sqrt_dim unit; do
        for tying_mode in ${WTE_TYING_MODES}; do
          case "${mode}" in
            unconstrained) fixed=false; radius="" ;;
            sqrt_dim) fixed=true; radius="" ;;
            unit) fixed=true; radius=1 ;;
          esac
          case "${tying_mode}" in
            tied) weight_tying=true ;;
            untied) weight_tying=false ;;
            *) echo "Unknown WTE tying mode: ${tying_mode}" >&2; exit 2 ;;
          esac

          name="dim-${embedding_dim}_digits-${num_digits}_letters-${num_letters}_${mode}_${tying_mode}"
          echo "=== ${name} ==="
          NUM_DIGITS="${num_digits}" \
          NUM_LETTERS="${num_letters}" \
          EMBEDDING_DIM="${embedding_dim}" \
          WTE_FIXED_NORM="${fixed}" \
          WTE_FIXED_NORM_VALUE="${radius}" \
          WTE_WEIGHT_TYING="${weight_tying}" \
          MAX_ITERS="${SWEEP_MAX_ITERS}" \
          SAVE_INTERVAL="${SWEEP_SAVE_INTERVAL}" \
          OUT_DIR="out/digits_3d_sweep/${name}" \
          TRAJECTORY_FILE="${RUNS_DIR}/${name}.json" \
            bash demos/digits_3d_trajectory_demo.sh
          python3 analysis/update_3d_sweep_manifest.py --runs-dir "${RUNS_DIR}"
        done
      done
    done
  done
done

cat <<EOF
Sweep complete. Serve the repository with:
  python3 -m http.server 8000

Example result:
  http://localhost:8000/report/threejs/digits-3d/index.html?data=runs/dim-3_digits-10_letters-10_sqrt_dim_tied.json
Sweep selector:
  http://localhost:8000/report/threejs/digits-3d/sweep.html
EOF
