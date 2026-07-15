#!/bin/bash
# Demo: train a small CoNaLa instruction-to-code model and sample with
# entropy/flatness-scaled dynamic temperature.
set -euo pipefail

if [ ! -f data/conala/input.txt ]; then
  echo "Preparing CoNaLa input.txt..."
  (cd data/conala && bash get_dataset.sh)
fi

python3 train.py \
  --dataset data/conala \
  --out_dir ./out_conala_dynamic_sampling \
  --max_iters 2000 \
  --eval_interval 500 \
  --log_interval 10 \
  --compile

python3 sample.py \
  --out_dir ./out_conala_dynamic_sampling \
  --start $'#U:\nwrite python code to flatten a list of lists\n#B:\n' \
  --max_new_tokens 160 \
  --num_samples 3 \
  --top_k 50 \
  --temperature 0.8 \
  --dynamic_temperature \
  --dynamic_temperature_min 0.35 \
  --dynamic_temperature_max 1.75 \
  --sample_file ./out_conala_dynamic_sampling/conala_dynamic_samples.txt
