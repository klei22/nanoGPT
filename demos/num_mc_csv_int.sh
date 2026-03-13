#!/usr/bin/env bash
# Integer-quantized CSV numerical multicontext demo (scalar mode):
# 1) build per-column integer datasets via shift/scale/round/clip
# 2) train numerical multicontext model with scalar input format
# 3) sample and write Plotly channel report

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CSV_INPUT="${1:-data/csv_num_mc_int/input.csv}"

data/csv_num_mc_int/get_datasets.sh "$CSV_INPUT" \
  --output_root csv_num_mc_int \
  --train_ratio 0.9 \
  --column-transform bpm:-30:250 \
  --column-transform spo2:-60:500 \
  --column-transform movement:0:200


python3 train.py \
  --training_mode multicontext \
  --dataset csv_num_mc_int/bpm \
  --multicontext \
  --multicontext_datasets \
    csv_num_mc_int/bpm \
    csv_num_mc_int/spo2 \
    csv_num_mc_int/movement \
  --numerical_multicontext \
  --numerical_multicontext_input_format scalar \
  --numerical_embedding_variant mlp \
  --numerical_output_variant mlp \
  --numerical_mlp_hidden_dims 128 128 128 \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --use_peri_ln \
  --attention_variant infinite \
  --use_concat_heads \
  --n_layer 8 \
  --n_head 3 \
  --n_qk_head_dim 200 \
  --n_v_head_dim 200 \
  --n_embd 512 \
  --block_size 256 \
  --batch_size 32 \
  --max_iters 3000 \
  --eval_interval 300 \
  --eval_iters 100 \
  --learning_rate 3e-4 \
  --dtype bfloat16 \
  --compile \
  --out_dir out/numerical_mc_csv_int

python3 sample.py \
  --out_dir out/numerical_mc_csv_int \
  --multicontext \
  --multicontext_datasets \
    csv_num_mc_int/bpm \
    csv_num_mc_int/spo2 \
    csv_num_mc_int/movement \
  --multicontext_start "11000" "18688" "800" \
  --numerical_multicontext_plotly \
  --numerical_multicontext_plotly_file out/numerical_mc_csv_int/num_mc_csv_int_samples.html \
  --max_new_tokens 256 \
  --num_samples 1
