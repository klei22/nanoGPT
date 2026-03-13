#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${1}" # mp3 wav or flac files
BASENAME="$(basename "$INPUT_FILE")"
STEM="${BASENAME%.*}"

python prepare.py \
  --method whisper_mel_csv \
  --train_input "$1" \
  --train_output "${STEM}.csv"
