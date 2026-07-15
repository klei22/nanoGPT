#!/bin/bash
# Demo: train a small CoNaLa instruction-to-code model, then compare baseline
# confidence heatmaps against dynamic-temperature effective colorization.
set -euo pipefail

OUT_DIR=${OUT_DIR:-./out_conala_dynamic_sampling}
PROMPT=${PROMPT:-$'#U:\nwrite python code to flatten a list of lists\n#B:\n'}
SEED=${SEED:-1337}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
TOP_K=${TOP_K:-50}
TEMPERATURE=${TEMPERATURE:-0.8}

if [ ! -f data/conala/input.txt ]; then
  echo "Preparing CoNaLa input.txt..."
  (cd data/conala && bash get_dataset.sh)
fi

python3 train.py \
  --dataset data/conala \
  --out_dir "$OUT_DIR" \
  --max_iters 2000 \
  --eval_interval 500 \
  --log_interval 10 \
  --compile

# Baseline: same seed/settings, no dynamic temperature. This emits the original
# per-token confidence colorization and heatmap images under OUT_DIR for comparison.
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --start "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_samples 1 \
  --top_k "$TOP_K" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --colorize_output \
  --colorize_mode softmax \
  --show_heatmaps \
  --sample_file "$OUT_DIR/conala_baseline_confidence.txt"

# Dynamic: same seed/settings plus dynamic temperature. The color map visualizes
# effective per-token temperature: green is lower/sharper, red is higher/flatter.
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --start "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_samples 1 \
  --top_k "$TOP_K" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --dynamic_temperature \
  --dynamic_temperature_min 0.35 \
  --dynamic_temperature_max 1.75 \
  --colorize_output \
  --colorize_mode dynamic_temperature \
  --sample_file "$OUT_DIR/conala_dynamic_temperature_map.txt"
