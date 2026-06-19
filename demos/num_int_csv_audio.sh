#!/usr/bin/env bash
# Audio-to-integer CSV numerical multicontext demo:
# 1) convert mel CSV bins into a 3-column headered CSV
# 2) quantize each column to uint16 datasets
# 3) train numerical multicontext model
# 4) sample and write Plotly report

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MEL_CSV_INPUT="${1:-data/audio/dummy_sine_mel.csv}"
AUDIO_INPUT_CSV="${2:-data/audio/audio_num_int_input.csv}"
OUTPUT_ROOT="${3:-csv_num_mc_int_audio}"

python3 data/audio/mel_csv_to_numint_csv.py \
  --input_csv "$MEL_CSV_INPUT" \
  --output_csv "$AUDIO_INPUT_CSV" \
  --bins "10,30,60"

python3 data/csv_num_mc_int/prepare_csv_int_multicontext.py \
  --input_csv "$AUDIO_INPUT_CSV" \
  --output_root "$OUTPUT_ROOT" \
  --train_ratio 0.9 \
  --column-transform mel_bin_010:0:1000 \
  --column-transform mel_bin_030:0:1000 \
  --column-transform mel_bin_060:0:1000

python3 train.py \
  --training_mode multicontext \
  --dataset "$OUTPUT_ROOT/mel_bin_010" \
  --multicontext \
  --multicontext_datasets \
    "$OUTPUT_ROOT/mel_bin_010" \
    "$OUTPUT_ROOT/mel_bin_030" \
    "$OUTPUT_ROOT/mel_bin_060" \
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
  --out_dir out/numerical_mc_csv_int_audio

python3 sample.py \
  --out_dir out/numerical_mc_csv_int_audio \
  --multicontext \
  --multicontext_datasets \
    "$OUTPUT_ROOT/mel_bin_010" \
    "$OUTPUT_ROOT/mel_bin_030" \
    "$OUTPUT_ROOT/mel_bin_060" \
  --multicontext_start_files \
    "data/$OUTPUT_ROOT/mel_bin_010/val.bin" \
    "data/$OUTPUT_ROOT/mel_bin_030/val.bin" \
    "data/$OUTPUT_ROOT/mel_bin_060/val.bin" \
  --multicontext_start_file_dtype uint16 \
  --multicontext_start_file_max_tokens 128 \
  --numerical_multicontext_plotly \
  --numerical_multicontext_plotly_file out/numerical_mc_csv_int_audio/num_int_csv_audio_samples.html \
  --max_new_tokens 256 \
  --num_samples 1
