#!/usr/bin/env bash
# Validate predictive-width components and optionally launch the paired smoke
# experiment. Usage:
#   demos/predictive_width_experiments.sh test   # unit tests only (default)
#   demos/predictive_width_experiments.sh smoke  # tests, data prep, experiment
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-test}"
case "$MODE" in
  test|smoke) ;;
  *) echo "usage: $0 [test|smoke]" >&2; exit 2 ;;
esac

echo "[predictive-width] running focused unit tests"
python3 -m pytest -q \
  tests/test_predictive_width.py \
  tests/test_predictive_width_metrics.py \
  tests/test_router_variations.py \
  tests/test_predictive_width_exploration.py

if [[ "$MODE" == "test" ]]; then
  exit 0
fi

DATA_DIR="$REPO_ROOT/data/shakespeare_char"
if [[ ! -f "$DATA_DIR/train.bin" || ! -f "$DATA_DIR/val.bin" || ! -f "$DATA_DIR/meta.pkl" ]]; then
  echo "[predictive-width] preparing Shakespeare character data"
  (
    cd "$DATA_DIR"
    if [[ ! -f input.txt ]]; then
      bash get_dataset.sh
    fi
    python3 prepare.py
  )
fi

echo "[predictive-width] launching dense/direct/collapsed paired smoke screen"
python3 optimization_and_search/run_experiments.py \
  --config explorations/predictive_width_smoke.yaml \
  --config_format yaml \
  --output_dir out_predictive_width_smoke \
  --prefix predictive_width_smoke
