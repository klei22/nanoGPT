#!/bin/bash

# Demo: Activation QAT with learned clipping and output kurtosis regularization.
# This runs a tiny model for a few iterations to validate the training flow.

python3 train.py \
  --out_dir "qat_kurtosis_demo" \
  --n_layer "2" \
  --n_head "2" \
  --n_kv_group "2" \
  --n_embd "64" \
  --max_iters "50" \
  --block_size "32" \
  --eval_iters "10" \
  --log_interval "10" \
  --dtype "bfloat16" \
  --quantization_warmup_iters 0 \
  --quantize_attn_act \
  --quantize_mlp_act \
  --quantize_attn_act_bits 4 \
  --quantize_mlp_act_bits 4 \
  --activations_quant_method "symmetric_quant" \
  --activation_qat \
  --activation_qat_clip_init 4.0 \
  --activation_kurtosis_reg 1e-5 \
  --activation_kurtosis_eps 1e-6
