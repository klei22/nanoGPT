#!/bin/bash

set -x

for (( i = 15; i < 26; i++ )); do
  python3 prepare.py \
    -t "input_${i}_$((i+1)).txt" \
    --method char \
    --reuse_chars \
    --train_output "train_${i}.bin" \
    --val_output "val_${i}.bin"
done
