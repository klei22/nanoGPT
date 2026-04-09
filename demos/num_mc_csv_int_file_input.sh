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
  --column-transform bpm:-30:250 \
  --column-transform spo2:-60:500 \
  --column-transform movement:0:200



for i in "0.1" "0.01" "0.001"; do
  for j in "0.1" "0.01" "0.001"; do
  python3 train.py \
    --training_mode multicontext \
    --dataset csv_num_mc_int/bpm \
    --numerical_loss_use_cosine \
    --numerical_loss_cosine_coeff "$j" \
    --multicontext \
    --multicontext_datasets \
      csv_num_mc_int/bpm \
      csv_num_mc_int/spo2 \
      csv_num_mc_int/movement \
    --numerical_multicontext \
    --numerical_loss_huber_delta "$i" \
    --numerical_multicontext_input_format scalar \
    --numerical_mlp_activation_variant gelu \
    --numerical_embedding_variant mlp \
    --numerical_output_variant mlp \
    --numerical_mlp_hidden_dims 128 128 128 \
    --use_qk_norm \
    --use_qk_norm_scale \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --attention_variant infinite \
    --use_concat_heads \
    --dropout 0.2 \
    --mlp_size 2000 \
    --n_layer 12 \
    --n_head 5 \
    --n_qk_head_dim 200 \
    --n_v_head_dim 200 \
    --n_embd 384 \
    --block_size 250 \
    --batch_size 64 \
    --max_iters 600 \
    --eval_interval 300 \
    --eval_iters 100 \
    --dtype float16 \
    --compile \
    --out_dir out/numerical_mc_csv_int_file_input_"$i"_"$j"

  # Use channel-specific binary token files as prompts; keep only the most recent
  # tokens to control prompt length.
  python3 sample.py \
    --out_dir out/numerical_mc_csv_int_file_input_"$i" \
    --multicontext \
    --multicontext_datasets \
      csv_num_mc_int/bpm \
      csv_num_mc_int/spo2 \
      csv_num_mc_int/movement \
    --multicontext_start_files \
      data/csv_num_mc_int/bpm/val.bin \
      data/csv_num_mc_int/spo2/val.bin \
      data/csv_num_mc_int/movement/val.bin \
    --multicontext_start_file_dtype uint16 \
    --multicontext_start_file_max_tokens 128 \
    --numerical_multicontext_plotly \
    --numerical_multicontext_plotly_file out/samples/num_mc_csv_int_file_input_samples_"$i"_"$j".html \
    --max_new_tokens 1024 \
    --num_samples 1
done
done
