#!/bin/bash

# Train a fully quantized model
## on the dataset tinystories
## using a linear quantization scheduler, increasing to full quantization
## after 45000 iterations
python3 train.py \
    --max_iters 90000 \
    --full_quant_iteration 45000 \
    --dataset tiny-stories \
    --n_head 8 \
    --n_embd 512 \
    --block_size 256 \
    --bias false \
    --dtype bfloat16 \
    --quantization_warmup_iters 0 \
    --quantize_attn_act true \
    --quantize_mlp_act true \
    --linear_variant_attn quantized_linear \
    --linear_variant_mlp quantized_linear \
    --quantize_linear_method symmetric_quant \
    --activations_quant_method symmetric_quant \
    --dropout 0 \
    --grad_clip 1.0 \
    --beta1 0.95 \
    --beta2 0.95 \
    --weight_decay 0.05 \
    --learning_rate 0.75e-3 \
    --quant_scheduler linear \
    --max_sample_tokens 100 \
    --sample_each_eval true

# Test the model's inference capabilities when holding the scales and zero points static
python3 sample.py \
    --out_dir quantization_tinystories/tiny_stories \
    --eval_only \
    --eval_dataset="tiny-stories" \
    --static_eval_scales