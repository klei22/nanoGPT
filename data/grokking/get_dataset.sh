#!/usr/bin/env bash
# data/grokking/get_dataset.sh

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 generate_grokking_data.py \
  --modulus 97 \
  --train_fraction 0.3 \
  --seed 1337 \
  --output_dir "$SCRIPT_DIR"

python3 prepare.py \
  --train_file "$SCRIPT_DIR/train.txt" \
  --val_file "$SCRIPT_DIR/val.txt" \
  --token_file "$SCRIPT_DIR/tokens.txt" \
  --output_dir "$SCRIPT_DIR"
