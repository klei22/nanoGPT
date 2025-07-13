#!/bin/bash

# Example training run using softmax-1 (implemented via MLA-LOBO) and OrthoAdam
# This is a short demo; adjust dataset and iterations for real training.
python3 train.py \
  --dataset openwebtext \
  --block_size 256 \
  --batch_size 8 \
  --max_iters 1000 \
  --eval_iters 100 \
  --eval_interval 200 \
  --optimizer orthoadam \
  --use_mla_lobo \
  --learning_rate 1e-3 \
  --beta1 0.9 \
  --beta2 0.999 \
  --weight_decay 0.1 \
  --ortho_perm_threshold 1000000 \
  --ortho_tiny_threshold 128 \
  --ortho_seed 1 \
  --out_dir orthoadam_demo
