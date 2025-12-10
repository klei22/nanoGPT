#!/bin/bash
# Demo: convert arbitrary audio files into fp16-packed datasets and launch numerical multicontext training.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 /path/to/audio1.wav [/path/to/audio2.flac ...]"
  exit 1
fi

# Minimal dependencies for audio loading
pip install --quiet numpy soundfile

AUDIO_OUTPUT_DIR="data/audio_fp16"

# 1) Preprocess the provided audio files
python data/audio_fp16/prepare_audio_fp16.py \
  --inputs "$@" \
  --output_dir "$AUDIO_OUTPUT_DIR" \
  --target_sample_rate 16000 \
  --val_fraction 0.1

# 2) Kick off a tiny numerical multi-context training run
python train.py \
  --training_mode multicontext \
  --dataset audio_fp16 \
  --multicontext \
  --multicontext_datasets audio_fp16 \
  --numerical_multicontext \
  --vocab_sizes 65536 \
  --block_size 256 \
  --batch_size 2 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 128 \
  --max_iters 20 \
  --eval_interval 5 \
  --out_dir out/audio_fp16_demo
