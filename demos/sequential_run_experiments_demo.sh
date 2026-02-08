#!/bin/bash
# sequential_run_experiments_demo.sh
# Demonstrates sequential experiment runs with resume support for
# train.py -> train_recurrent.py -> train_mezo.py using run_experiments.

set -euo pipefail

CONFIG_PATH="demos/sequential_run_experiments_demo.yaml"
OUT_DIR="out/sequential_run_demo"

mkdir -p "${OUT_DIR}"

echo "=== Running sequential experiment stages ==="
python3 optimization_and_search/run_experiments.py \
  --config "${CONFIG_PATH}" \
  --config_format yaml \
  --output_dir "${OUT_DIR}"

echo "Demo complete. Outputs are available under ${OUT_DIR}."
