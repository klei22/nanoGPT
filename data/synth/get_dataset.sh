#!/bin/bash
set -euo pipefail

# Download a subset of the PleIAs/SYNTH dataset by default.
# Pass --full to fetch every parquet shard exposed on the dataset page.

DATASET_URL="https://huggingface.co/datasets/PleIAs/SYNTH/tree/main"
MAX_FILES=2

if [[ "${1:-}" == "--full" || "${1:-}" == "--all" ]]; then
  echo "Downloading the full SYNTH dataset."
  MAX_FILES=""
else
  echo "Downloading the first ${MAX_FILES} parquet files from SYNTH. Use --full to fetch all shards."
fi

cmd=(
  python3 ../utils/get_parquet_dataset.py \
    --url "${DATASET_URL}" \
    --include_keys "query" "synthetic_reasoning" "synthetic_answer" \
    --value_prefix $'#Q:\n' $'#R:\n' $'#A:\n' \
    -o input.txt
)

if [[ -n "${MAX_FILES}" ]]; then
  cmd+=(--max_files "${MAX_FILES}")
fi

"${cmd[@]}"
