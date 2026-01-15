#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet/resolve/main/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data"
GLOBAL_TOTAL=10
LOCAL_TOTAL=10

usage() {
  cat <<'USAGE'
Usage: ./get_dataset.sh [options]

Options:
  --global-shard N   Process only global shard N (1-10). Default: all.
  --local-shard N    Process only local shard N (0-9). Default: all.
  --start-id N       Starting shard id (integer, e.g. 0). Default: 0.
  --end-id N         Ending shard id (inclusive). Default: auto-detect until missing.
  -o, --output FILE  Output text file (default: input.txt).
  -h, --help         Show this help.

Examples:
  # Download everything (auto-detect shard ranges per global/local)
  ./get_dataset.sh

  # Download a specific range from global shard 1, local shard 0
  ./get_dataset.sh --global-shard 1 --local-shard 0 --start-id 0 --end-id 99
USAGE
}

OUTPUT_FILE="input.txt"
GLOBAL_SHARD=""
LOCAL_SHARD=""
START_ID=0
END_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --global-shard)
      GLOBAL_SHARD="$2"
      shift 2
      ;;
    --local-shard)
      LOCAL_SHARD="$2"
      shift 2
      ;;
    --start-id)
      START_ID="$2"
      shift 2
      ;;
    --end-id)
      END_ID="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

format_global() {
  printf "%02d" "$1"
}

format_shard() {
  printf "%08d" "$1"
}

shard_url() {
  local global="$1"
  local local_shard="$2"
  local shard_id="$3"
  local global_fmt
  global_fmt=$(format_global "$global")
  printf "%s/global-shard_%s_of_%s/local-shard_%s_of_%s/shard_%s_processed.parquet" \
    "$BASE_URL" "$global_fmt" "$GLOBAL_TOTAL" "$local_shard" "$LOCAL_TOTAL" "$(format_shard "$shard_id")"
}

shard_exists() {
  local url="$1"
  local status
  status=$(curl -s -o /dev/null -w "%{http_code}" "$url")
  [[ "$status" == "200" ]]
}

emit_parquet_text() {
  local url="$1"
  local output="$2"

  python3 - "$url" "$output" <<'PY'
import sys
from pathlib import Path
import requests
import pandas as pd

url = sys.argv[1]
output = sys.argv[2]
filename = url.split("/")[-1]

download_dir = Path("downloaded_parquets")
download_dir.mkdir(exist_ok=True)
parquet_path = download_dir / filename

if not parquet_path.exists():
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with parquet_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

df = pd.read_parquet(parquet_path)
if "text" not in df.columns:
    raise ValueError(f"Expected 'text' column in {parquet_path}")

with open(output, "a", encoding="utf-8") as f:
    for value in df["text"].dropna():
        text = str(value).strip()
        if text:
            f.write(text + "\n")
PY
}

process_shard_range() {
  local global="$1"
  local local_shard="$2"
  local start_id="$3"
  local end_id="$4"
  local current_id

  if [[ -z "$end_id" ]]; then
    current_id="$start_id"
    while true; do
      local url
      url=$(shard_url "$global" "$local_shard" "$current_id")
      if ! shard_exists "$url"; then
        break
      fi
      echo "Downloading ${url}"
      emit_parquet_text "$url" "$OUTPUT_FILE"
      current_id=$((current_id + 1))
    done
  else
    for ((current_id=start_id; current_id<=end_id; current_id++)); do
      local url
      url=$(shard_url "$global" "$local_shard" "$current_id")
      if ! shard_exists "$url"; then
        echo "Skipping missing shard: ${url}"
        continue
      fi
      echo "Downloading ${url}"
      emit_parquet_text "$url" "$OUTPUT_FILE"
    done
  fi
}

main() {
  if [[ -n "$GLOBAL_SHARD" ]]; then
    global_start="$GLOBAL_SHARD"
    global_end="$GLOBAL_SHARD"
  else
    global_start=1
    global_end="$GLOBAL_TOTAL"
  fi

  if [[ -n "$LOCAL_SHARD" ]]; then
    local_start="$LOCAL_SHARD"
    local_end="$LOCAL_SHARD"
  else
    local_start=0
    local_end=$((LOCAL_TOTAL - 1))
  fi

  : > "$OUTPUT_FILE"

  for ((global=global_start; global<=global_end; global++)); do
    for ((local_shard=local_start; local_shard<=local_end; local_shard++)); do
      echo "Processing global shard ${global}/${GLOBAL_TOTAL}, local shard ${local_shard}/${LOCAL_TOTAL}"
      process_shard_range "$global" "$local_shard" "$START_ID" "$END_ID"
    done
  done
}

main
