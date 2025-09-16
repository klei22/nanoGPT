#!/bin/bash
# demos/ptq_demo.sh

# 1. Prepare shakespeare_char dataset
bash data/shakespeare_char/get_dataset.sh

QUANT="${1:-5}"

# 2. Train a larger model on shakespeare_char
out_dir="out_ptq_demo"
out_dir_after="out_ptq_demo_${QUANT}"
run_name_before="ptq_fp16"
run_name_after="ptq_int${QUANT}"
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "$out_dir" \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_peri_ln \
  --block_size 256 \
  --max_iters 1000 \
  --compile \
  --compute_model_stats \
  --print_model_stats_table "${run_name_before}.csv" \
  --tensorboard_run_name "$run_name_before"

# 3. Sample from the original model
python3 sample.py \
  --out_dir "$out_dir" \
  --num_samples 1 \
  --max_new_tokens 256 \
  --colorize_output \
  --start "To be " \
  --sample_file before_ptq.txt

# 4. Apply fake PTQ (quant-level-bit uniform)
python3 quantizations/ptq/fake_quantize_ckpt.py \
  "$out_dir" \
  --num_bits "${QUANT}" \
  --out_dir "${out_dir_after}"

# 5. Compute model stats after quantization
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${out_dir_after}" \
  --init_from resume \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --max_sample_tokens 256 \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_peri_ln \
  --block_size 256 \
  --max_iters 1000 \
  --compile \
  --compute_model_stats \
  --print_model_stats_table "${run_name_after}.csv" \
  --tensorboard_run_name "${run_name_after}"

# 6. Sample from the quantized model
python3 sample.py \
  --out_dir "${out_dir_after}" \
  --num_samples 1 \
  --max_new_tokens 256 \
  --colorize_output \
  --start "To be " \
  --sample_file "after_ptq${QUANT}".txt

