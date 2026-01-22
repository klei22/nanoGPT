#!/bin/bash

BASE_URL="https://huggingface.co/datasets/cis-lmu/GlotCC-V1/resolve/main/v1.0/kor-Hang/kor-Hang_"
START_SHARD=0
END_SHARD=0      # This is the last file index (inclusive)

python3 ./utils/get_parquet_dataset_range.py \
  --url_base "${BASE_URL}" \
  --start_num ${START_SHARD} \
  --stop_num ${END_SHARD} \
  --total_shards "-1" \
  --padding_digits 1 \
  -i "content" \
  -p "" \
  -o "input.txt" \
  -s
