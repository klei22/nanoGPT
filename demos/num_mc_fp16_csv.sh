#!/usr/bin/env bash
# FP16-bits CSV numerical multicontext demo:
# 1) build per-column fp16-bit datasets from a CSV file
# 2) train numerical multicontext with model-side fp16 bit decoding
# 3) sample and write Plotly channel report

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Build data/csv_num_mc/<column>/train.bin,val.bin,meta.pkl
# Use a real CSV path as first arg, or fallback to template input.csv.
CSV_INPUT="${1:-data/csv_num_mc/input.csv}"

data/csv_num_mc/get_datasets.sh "$CSV_INPUT" \
  --output_root csv_num_mc \
  --train_ratio 0.9

# Example uses two contexts: bpm + spo2.
python3 train.py \
  --training_mode multicontext \
  --dataset csv_num_mc/bpm \
  --multicontext \
  --multicontext_datasets \
    csv_num_mc/bpm \
    csv_num_mc/spo2 \
  --numerical_multicontext \
  --numerical_multicontext_input_format fp16_bits \
  --numerical_embedding_variant mlp \
  --numerical_output_variant mlp \
  --numerical_mlp_hidden_dims 128 128 \
  --norm_channel_variant hyperspherenorm \
  --norm_channel_radius 1.0 \
  --norm_channel_scale 1.0 \
  --no-norm_channel_gain \
  --no-norm_channel_radius_learning \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --block_size 256 \
  --batch_size 32 \
  --max_iters 3000 \
  --eval_interval 300 \
  --eval_iters 100 \
  --learning_rate 3e-4 \
  --dtype bfloat16 \
  --compile \
  --out_dir out/numerical_mc_fp16_csv

python3 sample.py \
  --out_dir out/numerical_mc_fp16_csv \
  --multicontext \
  --multicontext_datasets \
    csv_num_mc/bpm \
    csv_num_mc/spo2 \
  --multicontext_start "0" "0" \
  --numerical_multicontext_plotly \
  --numerical_multicontext_plotly_file out/numerical_mc_fp16_csv/num_mc_fp16_csv_samples.html \
  --max_new_tokens 256 \
  --num_samples 3
