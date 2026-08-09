#!/bin/bash

# Train a small model
python3 train.py \
  --out_dir ptq_demo_model \
  --n_layer 2 \
  --n_head 2 \
  --n_kv_group 2 \
  --n_embd 60 \
  --max_iters 100 \
  --block_size 32 \
  --eval_iters 50 \
  --log_interval 20

# Post-training quantize the checkpoint using KurTail
python3 quantization/ptq_quantize_ckpt.py \
  --ckpt_path ptq_demo_model/ckpt.pt \
  --out_ckpt ptq_demo_model/ckpt_ptq.pt \
  --bits 4 \
  --quant_method kurtail_quant

# Run inference from the quantized checkpoint
cp ptq_demo_model/ckpt_ptq.pt ptq_demo_model/ckpt.pt
python3 sample.py \
  --out_dir ptq_demo_model \
  --init_from resume \
  --start "Hello" \
  --num_samples 1 \
  --max_new_tokens 50
