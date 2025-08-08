#!/bin/bash
# demos/ngpt_demo.sh
# Minimal example of training with nGPT normalization utilities

# Ensure Shakespeare dataset is present
bash data/shakespeare_char/get_dataset.sh

# Train a tiny model with nGPT enabled
python3 train.py \
  --dataset shakespeare_char \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --block_size 64 \
  --max_iters 200 \
  --eval_iters 100 \
  --log_interval 10 \
  --out_dir out-ngpt-demo \
  --use_ngpt
