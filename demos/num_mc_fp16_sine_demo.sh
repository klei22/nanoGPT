#!/bin/bash
set -euo pipefail

# End-to-end demo:
# 1) create multi-context sine datasets where each token is an fp16 bit pattern (uint16)
# 2) train numerical multicontext model that decodes these bits to fp32 values in model.py
# 3) sample predictions and decode generated tokens back to fp32 values

python3 data/sinewave/create_fp16_sine_multicontext.py \
  --dataset_root data/sinewave_fp16 \
  --num_contexts 8 \
  --points_per_context 32768 \
  --train_split 0.9 \
  --period 64.0

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
  --numerical_input_token_format fp16_bits \
  --numerical_mlp_hidden_dim 96 \
  --numerical_mlp_num_layers 3 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 256 \
  --block_size 256 \
  --batch_size 24 \
  --max_iters 3000 \
  --eval_interval 500 \
  --compile \
  --out_dir out/numerical_mc_sine_fp16

python3 sample.py \
  --out_dir out/numerical_mc_sine_fp16 \
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
  --multicontext_start \
    "0.000000,0.098145,0.191406,0.277832" \
    "0.000000,0.194946,0.353516,0.447754" \
    "0.000000,0.287109,0.530273,0.690430" \
    "0.000000,0.371094,0.667969,0.833984" \
    "0.000000,0.442383,0.759766,0.890625" \
    "0.000000,0.301758,0.540039,0.697754" \
    "0.000000,0.201904,0.365967,0.468018" \
    "0.000000,0.117676,0.216064,0.289307" \
  --max_new_tokens 128 \
  --num_samples 1
