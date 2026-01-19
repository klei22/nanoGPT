#!/bin/bash

# Downloads the Europarl en-pt parquet shards from Hugging Face and emits the
# translation fields to input.txt.

set -euo pipefail

url="https://huggingface.co/datasets/Helsinki-NLP/europarl/tree/main/en-pt"

python3 ./utils/get_parquet_translation_dataset.py \
  --url "${url}" \
  --language_keys "en" "pt" \
  --value_prefix $'\n#EN:\n' $'\n#PT:\n' \
  --output_text_file "input.txt" \
  --skip_empty
