#!/bin/bash

dataset="minipile"
# obtain and tokenize minipile
pushd data/"$dataset"
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for ${dataset}"
fi
popd

# Train a fully quantized asic model
## using a linear quantization scheduler, increasing to full quantization
## after 45000 iterations
python3 train.py \
    --use_edgellm_asic true \
    --max_iters 90000 \
    --full_quant_iteration 45000 \
    --dataset "$dataset" \
    --n_head 8 \
    --n_embd 512 \
    --block_size 256 \
    --batch_size 16 \
    --bias false \
    --dtype bfloat16 \
    --quantization_warmup_iters 0 \
    --use_pre_ln true \
    --quantize_attn_act true \
    --quantize_mlp_act true \
    --quantize_asic_prenorm true \
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
    --out_dir quantization_"$dataset"/"$dataset" \
    --eval_only \
    --eval_dataset="$dataset" \
    --static_eval_scales
