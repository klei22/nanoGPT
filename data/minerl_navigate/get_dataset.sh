#!/bin/bash
# Helper wrapper to generate pixel-wise channel arrays for the MineRL Navigate dataset.
#
# Customize OUTPUT_DIR or SPLIT via environment variables when invoking the script:
#   OUTPUT_DIR=/tmp/minerl_pixels SPLIT=test bash get_dataset.sh

set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-"./minerl_navigate_output"}
DATASET_DIR=${DATASET_DIR:-"${HOME}/.cache/minerl_navigate"}
SPLIT=${SPLIT:-"train"}
MAX_VIDEOS=${MAX_VIDEOS:-""}

# Ensure dependencies are present
python - <<'PY'
import importlib
for pkg in ["imageio"]:
    importlib.import_module(pkg)
PY

echo "Writing ${SPLIT} split pixel channels to ${OUTPUT_DIR}"

python "$(dirname "$0")/extract_minerl_navigate_pixels.py" \
  --split "${SPLIT}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  ${MAX_VIDEOS:+--max_videos "${MAX_VIDEOS}"}
