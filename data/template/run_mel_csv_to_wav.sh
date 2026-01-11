#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_mel_csv_to_wav.sh mel.csv reconstructed.wav

CSV_PATH="${1:-mel.csv}"
OUT_WAV="${2:-reconstructed.wav}"

python3 mel_csv_to_wav.py "${CSV_PATH}" \
  --output "${OUT_WAV}" \
  --sample_rate 16000 \
  --n_fft 400 \
  --hop_length 160 \
  --win_length 400 \
  --n_mels 80 \
  --f_min 0 \
  --f_max 8000 \
  --center \
  --power 2.0 \
  --normalized_input \
  --griffin_lim_iters 32
