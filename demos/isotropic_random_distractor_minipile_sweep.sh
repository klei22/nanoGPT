#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-explorations/isotropic_random_distractor_minipile_train.yaml}"
OUT_ROOT="${2:-out_isotropic_minipile}"
PREFIX="${3:-ird_minipile}"
REPORT_DIR="${4:-report/isotropic_random_distractor_minipile}"
LOG_FILE="$OUT_ROOT/$OUT_ROOT.yaml"

cat <<EOF
=== Isotropic Random-Distractor minipile sweep ===
Config:      $CONFIG
Output root: $OUT_ROOT
Log file:    $LOG_FILE
Report dir:  $REPORT_DIR
EOF

if [ ! -f data/minipile/train.bin ] || [ ! -f data/minipile/val.bin ] || [ ! -f data/minipile/meta.pkl ]; then
  echo "Preparing minipile dataset..."
  (cd data/minipile && bash get_dataset.sh && if [ -f prepare.py ]; then python3 prepare.py -t input.txt --method tiktoken; fi)
fi

if [ ! -f data/minipile/train.bin ] || [ ! -f data/minipile/val.bin ] || [ ! -f data/minipile/meta.pkl ]; then
  echo "Minipile tokenized artifacts are still missing after preparation." >&2
  echo "If get_dataset.sh downloaded input.txt but no prepare.py is present, copy the repo tokenizer prep script used by your minipile setup and re-run." >&2
  exit 1
fi

python3 optimization_and_search/run_from_yaml.py \
  --yaml "$CONFIG" \
  --output_dir "$OUT_ROOT" \
  --prefix "$PREFIX" \
  --dataset minipile

if [ ! -f "$LOG_FILE" ]; then
  echo "Could not find expected log file: $LOG_FILE" >&2
  echo "Search under $OUT_ROOT for the generated YAML log, then run:" >&2
  echo "  python3 analysis/isotropic_random_distractor/analyze_minipile_sweep.py --log <log.yaml> --outdir $REPORT_DIR" >&2
  exit 1
fi

python3 analysis/isotropic_random_distractor/analyze_minipile_sweep.py \
  --log "$LOG_FILE" \
  --outdir "$REPORT_DIR"

cat <<EOF

Minipile isotropic sweep complete.
Open:
  $REPORT_DIR/index.html
EOF
