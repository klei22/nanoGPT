#!/bin/bash

# Download the GSM-Symbolic dataset (apple/GSM-Symbolic) JSONL files and
# convert them into the plain-text `input.txt` format expected by the
# template/prepare.py pipeline.

set -euo pipefail

BASE_URL="https://huggingface.co/datasets/apple/GSM-Symbolic/resolve/main"
DOWNLOAD_DIR="downloaded_jsons"
OUTPUT_JSON_DIR="json_output"

mkdir -p "${DOWNLOAD_DIR}" "${OUTPUT_JSON_DIR}"

# The dataset is split across three shards under different folders.
SPLITS=(
  "main/test.jsonl"
  "p1/test.jsonl"
  "p2/test.jsonl"
)

for split_path in "${SPLITS[@]}"; do
  # Replace path separators so we can keep the files flat under DOWNLOAD_DIR.
  split_slug=${split_path//\//_}
  target_path="${DOWNLOAD_DIR}/${split_slug}"

  if [ ! -f "${target_path}" ]; then
    echo "Downloading ${split_path}..."
    curl -L "${BASE_URL}/${split_path}" -o "${target_path}"
  else
    echo "Found existing ${target_path}, skipping download"
  fi
done

# Combine the JSONL shards into a single JSON array for downstream processing.
python - <<'PY'
import json
from pathlib import Path

download_dir = Path("downloaded_jsons")
combined_path = Path("json_output/combined.json")
combined_path.parent.mkdir(exist_ok=True)

records = []
for jsonl_path in sorted(download_dir.glob("*.jsonl")):
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)

combined_path.write_text(json.dumps(records))
print(f"Wrote {len(records)} records to {combined_path}")
PY

# Emit question/answer pairs with explicit user/bot prefixes.
python ./utils/get_json_dataset.py \
  --direct_json_input ${OUTPUT_JSON_DIR}/combined.json \
  --output_text_file input.txt \
  --include_keys "question" "answer" \
  --value_prefixes $'#U:\n' $'#B:\n' \
  --skip_empty

# Normalize final-answer markers to a more descriptive prefix.
sed -i "s/^####/answer:/" input.txt
