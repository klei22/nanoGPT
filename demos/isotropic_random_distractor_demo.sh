#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-explorations/isotropic_random_distractor.yaml}"
OUTDIR="${2:-report/isotropic_random_distractor}"

python3 analysis/isotropic_random_distractor/run_experiments.py \
  --config "$CONFIG" \
  --outdir "$OUTDIR"

cat <<EOF

Isotropic Random-Distractor demo complete.
Open the self-contained report at:
  $OUTDIR/index.html

Useful quick run:
  bash demos/isotropic_random_distractor_demo.sh explorations/isotropic_random_distractor_fast.yaml report/isotropic_random_distractor_fast
EOF
