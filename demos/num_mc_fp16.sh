#!/bin/bash
set -euo pipefail

# End-to-end numerical multi-context demo using fp16 bit-pattern encoded sine waves.
# This demonstrates two paths:
#   1) Existing raw numeric-bin mapping (numerical_input_encoding=raw)
#   2) New fp16-bit mapping decoded in model.py via _fp16bits_to_fp32
#
# Usage:
#   bash demos/num_mc_fp16.sh fp16_bits   # default behavior
#   bash demos/num_mc_fp16.sh raw         # fallback to legacy numeric mapping behavior

ENCODING_MODE="${1:-fp16_bits}"
if [[ "$ENCODING_MODE" != "fp16_bits" && "$ENCODING_MODE" != "raw" ]]; then
  echo "Unsupported encoding mode: $ENCODING_MODE"
  echo "Valid choices: fp16_bits | raw"
  exit 1
fi

OUT_DIR="out/numerical_mc_sine_${ENCODING_MODE}"

if [[ "$ENCODING_MODE" == "fp16_bits" ]]; then
  echo "[1/3] Creating fp16-bit sine datasets (different amplitudes/phases/periods)."
  bash data/sinewave_fp16/get_dataset.sh \
    --num_contexts 8 \
    --points_per_period 64 \
    --num_periods 800 \
    --base_amplitude 0.5 \
    --amplitude_step 0.1 \
    --phase_step 0.4 \
    --base_period 1.0 \
    --period_step 0.12

  DATASETS=(
    sinewave_fp16/s1
    sinewave_fp16/s2
    sinewave_fp16/s3
    sinewave_fp16/s4
    sinewave_fp16/s5
    sinewave_fp16/s6
    sinewave_fp16/s7
    sinewave_fp16/s8
  )
else
  echo "[1/3] Using existing integer sinewave bins (legacy mapping)."
  bash data/sinewave/get_dataset.sh
  DATASETS=(
    sinewave/s1
    sinewave/s8
    sinewave/s11
    sinewave/s13
    sinewave/s14
  )
fi

echo "[2/3] Training numerical multi-context model with encoding=$ENCODING_MODE"
python train.py \
  --training_mode multicontext \
  --dataset "${DATASETS[0]}" \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --numerical_multicontext \
  --numerical_input_encoding "$ENCODING_MODE" \
  --numerical_mlp_hidden_dim 64 \
  --n_layer 8 \
  --n_head 4 \
  --n_embd 256 \
  --block_size 256 \
  --batch_size 32 \
  --max_iters 2000 \
  --eval_interval 250 \
  --out_dir "$OUT_DIR"

echo "[3/3] Sampling from trained checkpoint"
python sample.py \
  --out_dir "$OUT_DIR" \
  --init_from resume \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --numerical_multicontext \
  --numerical_input_encoding "$ENCODING_MODE" \
  --start "" \
  --max_new_tokens 96 \
  --num_samples 1
