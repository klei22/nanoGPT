#!/bin/bash

# Downloads the parquet shards from Hugging Face and emits text_eng* columns to input.txt
# You can modify INCLUDE_KEYS to pull different language columns from the schema.

set -euo pipefail

# Base URL for the dataset's parquet files
url="https://huggingface.co/datasets/muhammadravi251001/restructured-ntrex128/tree/main/data"

python3 ../template/utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "text_eng_Latn" "text_eng-US_Latn" "text_eng-GB_Latn" \
  --value_prefix $'#TEXT:\n' $'#TEXT:\n' $'#TEXT:\n' \
  --output_text_file "input.txt" \
  --skip_empty
