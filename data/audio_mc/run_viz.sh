#!/usr/bin/env bash
set -euo pipefail

mkdir -p images
pre_dot_name="${1%%.*}"

python3 sample_gen_utils/visualize_whisper_mel_csv.py "${pre_dot_name}.csv" --output "images/${pre_dot_name}.png"

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "images/${pre_dot_name}.png" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "images/${pre_dot_name}.png" || true
fi

