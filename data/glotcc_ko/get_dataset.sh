#!/bin/bash

url="https://huggingface.co/datasets/cis-lmu/GlotCC-V1/tree/main/v1.0/kor-Hang"

python3 ./utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "content" \
  --value_prefix $'\n'

