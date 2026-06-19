#!/bin/bash

# Example Quantization Aware Training using KurTail rotation
python3 train.py \
  --out_dir "kurtail_qat_model" \
  --n_layer "2" \
  --n_head "2" \
  --n_kv_group "2" \
  --n_embd "60" \
  --max_iters "100" \
  --block_size "32" \
  --eval_iters "50" \
  --log_interval "20" \
  --quantize_linear_method "kurtail_quant" \
  --activations_quant_method "kurtail_quant" \
  --dtype "bfloat16" \
  --quantization_warmup_iters 0 \
  --quantize_attn_act \
  --quantize_mlp_act \
  --linear_variant_attn "quantized_linear" \
  --linear_variant_mlp "quantized_linear" \
  --store_activations

