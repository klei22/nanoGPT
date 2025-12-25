#!/usr/bin/env bash
set -euo pipefail

python3 sample_gen_utils/mel.py \
  --mode mel_bin \
  --sr 16000 \
  --duration 10 \
  --triangle-period 2.0 \
  --mel-n-mels 80 \
  --mel-f-min 0 \
  --mel-f-max 8000 \
  --bin-low 10 \
  --bin-high 40 \
  --amp 0.5 \
  --out mel_10_to_70_triangle.wav

python3 prepare.py --method whisper_mel_csv \
  --train_input mel_10_to_70_triangle.wav \
  --train_output mel.csv \
  --mel_sample_rate 16000 \
  --mel_n_fft 400 \
  --mel_hop_length 160 \
  --mel_win_length 400 \
  --mel_n_mels 80 \
  --mel_f_min 0 \
  --mel_f_max 8000 \
  --mel_center \
  --mel_power 2.0 \
  --mel_normalize \
  --mel_csv_float_format "%.6f"

bash run_viz.sh mel.csv

