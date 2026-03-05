#!/usr/bin/env bash
# FP16-bits numerical multicontext demo:
# 1) generate sinewave contexts with regular sinewave-like phase/range and fp16-bit encoding
# 2) train numerical multicontext with model-side fp16 bit decoding

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 data/sinewave/create_fp16_multicontext_dataset.py \
  --output_root sinewave_fp16 \
  --contexts 8 \
  --train_ratio 0.9 \
  --base_period 15 \
  --period_step 1 \
  --base_phase 0.0 \
  --phase_step 0.0 \
  --base_amplitude 50 \
  --amplitude_step 0.0 \
  --points_per_period 15 \
  --num_periods 2000 \
  --dc_offset 64

python3 train.py \
  --training_mode multicontext \
  --dataset sinewave_fp16/s1 \
  --multicontext \
  --multicontext_datasets \
    sinewave_fp16/s1 \
    sinewave_fp16/s2 \
    sinewave_fp16/s3 \
    sinewave_fp16/s4 \
    sinewave_fp16/s5 \
    sinewave_fp16/s6 \
    sinewave_fp16/s7 \
    sinewave_fp16/s8 \
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
  --out_dir out/numerical_mc_fp16_sine
