#!/bin/bash

# Quantization aware training with KurTail on the Minipile dataset
python3 train.py \
  --dataset minipile \
  --out_dir kurtail_minipile \
  --n_layer 2 \
  --n_head 2 \
  --n_kv_group 2 \
  --n_embd 60 \
  --block_size 32 \
  --max_iters 100 \
  --eval_iters 50 \
  --log_interval 20 \
  --quantization_warmup_iters 0 \
  --quantize_attn_act \
  --quantize_mlp_act \
  --linear_variant_attn quantized_linear \
  --linear_variant_mlp quantized_linear \
  --quantize_linear_method kurtail_quant \
  --activations_quant_method kurtail_quant \
  --store_activations
