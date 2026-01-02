#!/bin/bash
# char_bpe_vocab_demo.sh
# Demonstrates Char-BPE tokenization with byte fallback, token-count tracking,
# and interactive vocabulary exploration.

set -euo pipefail

DATA_DIR="data/template"
WORK_DIR="${DATA_DIR}/demo_char_bpe"

mkdir -p "${WORK_DIR}"

cat <<'TEXT' > "${WORK_DIR}/input.txt"
Hello 👋! This is a small demo for Char-BPE + byte fallback.
It includes emoji, punctuation, and mixed scripts: 你好, Привет, مرحبا.
TEXT

pushd "${WORK_DIR}" > /dev/null

python3 ../prepare.py \
  -t input.txt \
  --method char_bpe \
  --vocab_size 128 \
  -T \
  --train_output train.bin \
  --val_output val.bin

echo "Launching the Char-BPE vocabulary explorer (press q to quit)."
python3 ../utils/explore_char_bpe_vocab.py \
  --vocab char_bpe_vocab.json \
  --counts char_bpe_token_counts.json

popd > /dev/null
