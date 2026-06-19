#!/bin/bash

cd ../

DATASET="shakespeare_char"
bash "data/${DATASET}/get_dataset.sh"

python3 train.py \
  --out_dir out_seq_with_skips \
  --device cpu \
  --eval_interval 2 \
  --log_interval 1 \
  --block_size 8 \
  --batch_size 2 \
  --n_layer 2 \
  --n_head 2 \
  --n_kv_group 2 \
  --n_embd 16 \
  --max_iters 3 \
  --lr_decay_iters 2 \
  --dropout 0.0 \
  --dataset "$DATASET" \
  --use_block_operation_sequence \
  --block_operation_sequence attn attn mlp mlp \
  --block_sequence_use_intermediate_skips

python3 train.py \
  --out_dir out_seq_outer_skip \
  --device cpu \
  --eval_interval 2 \
  --log_interval 1 \
  --block_size 8 \
  --batch_size 2 \
  --n_layer 2 \
  --n_head 2 \
  --n_kv_group 2 \
  --n_embd 16 \
  --max_iters 3 \
  --lr_decay_iters 2 \
  --dropout 0.0 \
  --dataset "$DATASET" \
  --use_block_operation_sequence \
  --block_operation_sequence attn mlp mlp mlp attn mlp \
  --no-block_sequence_use_intermediate_skips
