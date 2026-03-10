#!/usr/bin/env bash
# Integer-quantized CSV numerical multicontext demo with file-based prompt input:
# 1) build per-column integer datasets via shift/scale/round/clip
# 2) train numerical multicontext model with scalar input format
# 3) sample using per-channel .bin files instead of --multicontext_start strings

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CSV_INPUT="${1:-data/csv_num_mc_int/input.csv}"

data/csv_num_mc_int/get_datasets.sh "$CSV_INPUT" \
  --output_root csv_num_mc_int \
  --train_ratio 0.9 \
  --column-transform bpm:-40:2 \
  --column-transform spo2:0:10

python3 train.py \
  --training_mode multicontext \
  --dataset csv_num_mc_int/bpm \
  --multicontext \
  --multicontext_datasets \
    csv_num_mc_int/bpm \
    csv_num_mc_int/spo2 \
  --numerical_multicontext \
  --numerical_multicontext_input_format scalar \
  --numerical_embedding_variant mlp \
  --numerical_output_variant mlp \
  --numerical_mlp_hidden_dims 128 128 \
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
  --out_dir out/numerical_mc_csv_int_file_input

# Use channel-specific binary token files as prompts; keep only the most recent
# tokens to control prompt length.
python3 sample.py \
  --out_dir out/numerical_mc_csv_int_file_input \
  --multicontext \
  --multicontext_datasets \
    csv_num_mc_int/bpm \
    csv_num_mc_int/spo2 \
  --multicontext_start_files \
    data/csv_num_mc_int/bpm/train.bin \
    data/csv_num_mc_int/spo2/train.bin \
  --multicontext_start_file_dtype uint16 \
  --multicontext_start_file_max_tokens 128 \
  --numerical_multicontext_plotly \
  --numerical_multicontext_plotly_file out/numerical_mc_csv_int_file_input/num_mc_csv_int_file_input_samples.html \
  --max_new_tokens 256 \
  --num_samples 3
