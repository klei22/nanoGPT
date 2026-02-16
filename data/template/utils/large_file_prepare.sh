#!/bin/bash

input_file=${1:-input.txt}
tokenization=${2:-tiktoken}
char_bpe_vocab_path=${3:-}

python3 ./utils/partition_file.py --input_file "${input_file}"

batch_prepare_args=(
  --input_dir partitioned_file
  --prepare_script prepare.py
  --tokenizer "$tokenization"
)

if [[ "$tokenization" == "char_bpe" && -n "$char_bpe_vocab_path" ]]; then
  batch_prepare_args+=(--char_bpe_vocab_path "$char_bpe_vocab_path")
fi

python3 ./utils/batch_prepare.py "${batch_prepare_args[@]}"
