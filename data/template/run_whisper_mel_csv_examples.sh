#!/usr/bin/env bash
set -euo pipefail

# Example usage for Whisper-style mel CSV tokenizer with different audio formats.
# Adjust AUDIO_DIR to point at your audio files.

AUDIO_DIR="${AUDIO_DIR:-.}"

python data/template/prepare.py \
  --method whisper_mel_csv \
  --train_input "${AUDIO_DIR}/sample.mp3" \
  --train_output "sample_mp3.csv"

python data/template/prepare.py \
  --method whisper_mel_csv \
  --train_input "${AUDIO_DIR}/sample.wav" \
  --train_output "sample_wav.csv"

python data/template/prepare.py \
  --method whisper_mel_csv \
  --train_input "${AUDIO_DIR}/sample.flac" \
  --train_output "sample_flac.csv"
