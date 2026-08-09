#!/bin/bash

set -euo pipefail

declare -A url_array
url_array["en-ja"]="https://huggingface.co/datasets/haoranxu/FLORES-200/tree/main/en-ja"
url_array["en-ko"]="https://huggingface.co/datasets/haoranxu/FLORES-200/tree/main/en-ko"
url_array["en-zh"]="https://huggingface.co/datasets/haoranxu/FLORES-200/tree/main/en-zh"

pairs=("en-ja" "en-ko" "en-zh")

for pair in "${pairs[@]}"; do
  url="${url_array[$pair]}"
  src_lang="${pair%-*}"
  tgt_lang="${pair#*-}"

  python3 ./utils/get_parquet_dataset.py \
    --url "$url" \
    --include_keys "$src_lang" "$tgt_lang" \
    --value_prefix $"#${src_lang^^}:\n" $"#${tgt_lang^^}:\n" \
    --output_text_file "${pair}.txt" \
    --skip_empty
done

rm -f input.txt
for pair in "${pairs[@]}"; do
  cat "${pair}.txt" >> input.txt
  printf "\n#pair: %s\n\n" "$pair" >> input.txt
done
