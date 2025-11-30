#!/bin/bash
set -euo pipefail

# Download a subset of the PleIAs/SYNTH parquet dataset.
# By default, only the first two parquet files are processed to keep the
# download light-weight. Pass --full to include every parquet file listed
# on the dataset page.

DATA_URL="https://huggingface.co/datasets/PleIAs/SYNTH/tree/main"
OUTPUT_TEXT_FILE="input.txt"
INCLUDE_KEYS=("query" "synthetic_answer")
VALUE_PREFIXES=($'#Q:\n' $'#A:\n')

usage() {
  cat <<USAGE
Usage: bash get_dataset.sh [--full]

Options:
  --full, -f   Download and process every parquet file listed at ${DATA_URL}.
USAGE
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="partial"
if [[ "${1-}" == "--full" || "${1-}" == "-f" ]]; then
  MODE="full"
fi

echo "Using SYNTH dataset URL: ${DATA_URL}"
echo "Writing merged text to: ${OUTPUT_TEXT_FILE}"
python3 - "$MODE" "$DATA_URL" "$OUTPUT_TEXT_FILE" "${INCLUDE_KEYS[@]}" "${VALUE_PREFIXES[@]}" <<'PY'
import sys
from pathlib import Path
from utils import get_parquet_dataset as gpd

mode, url, output_text_file, *rest = sys.argv[1:]
include_keys = rest[:2]
value_prefixes = rest[2:]

parquet_links = gpd.find_parquet_links(url)
if not parquet_links:
    raise SystemExit(f"No parquet links found at {url}")

if mode != "full":
    parquet_links = parquet_links[:2]
    print(f"Downloading first {len(parquet_links)} parquet files (partial mode).")
else:
    print(f"Downloading all {len(parquet_links)} parquet files (full mode).")

download_dir = Path("downloaded_parquets")
json_dir = Path("json_output")
download_dir.mkdir(exist_ok=True)
json_dir.mkdir(exist_ok=True)

Path(output_text_file).write_text("")

for link in parquet_links:
    file_name = link.split("/")[-1].split("?")[0]
    parquet_path = download_dir / file_name
    json_path = json_dir / file_name.replace(".parquet", ".json")

    if not parquet_path.exists():
        gpd.download_file(link, parquet_path)
    else:
        print(f"{parquet_path} already exists, skipping download")

    gpd.convert_to_json(parquet_path, json_path)
    gpd.emit_json_contents(
        json_path,
        output_text_file,
        include_keys,
        value_prefixes,
        required_key=None,
        skip_empty=False,
        exclude=None,
        list_key=None,
        role_prefixes=None,
    )

print(f"Wrote merged text to {output_text_file}")
PY
