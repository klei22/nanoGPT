#!/bin/bash
# Demo script for numerical multi-context training with sine wave data.
# 1) build scalar sinewave datasets
# 2) train numerical multicontext model
# 3) sample and write Plotly channel report

set -euo pipefail

pushd data/sinewave/
if [[ ! -f "s1/train.bin" ]]; then
  bash get_dataset.sh
fi
popd

python train.py \
  --training_mode multicontext \
  --dataset sinewave/s1 \
  --multicontext \
  --multicontext_datasets \
    sinewave/s1 \
    sinewave/s8 \
    sinewave/s11 \
    sinewave/s13 \
    sinewave/s14 \
  --numerical_multicontext \
  --numerical_multicontext_input_format scalar \
  --numerical_mlp_hidden_dim 64 \
  --n_layer 10 \
  --n_head 6 \
  --n_embd 384 \
  --block_size 256 \
  --batch_size 32 \
  --max_iters 3000 \
  --eval_interval 500 \
  --compile \
  --out_dir out/numerical_mc_test_2

python sample.py \
  --out_dir out/numerical_mc_test_2 \
  --multicontext \
  --multicontext_datasets \
    sinewave/s1 \
    sinewave/s8 \
    sinewave/s11 \
    sinewave/s13 \
    sinewave/s14 \
  --multicontext_start "64" "64" "64" "64" "64" \
  --numerical_multicontext_plotly \
  --numerical_multicontext_plotly_file out/numerical_mc_test_2/num_mc_scalar_samples.html \
  --max_new_tokens 256 \
  --num_samples 3
