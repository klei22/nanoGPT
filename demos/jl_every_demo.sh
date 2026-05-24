#!/bin/bash
# demos/jl_every_demo.sh
# Demonstrates periodic Johnson–Lindenstrauss transforms during training

# Ensure Shakespeare dataset is present
bash data/shakespeare_char/get_dataset.sh

# Train a tiny model applying JL transform every 100 iterations
python3 train.py \
  --dataset shakespeare_char \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --block_size 64 \
  --max_iters 200 \
  --eval_iters 100 \
  --log_interval 10 \
  --out_dir out-jl-demo \
  --jl_every 100 \
  --jl_out_embd 48 \
  --jl_type gaussian
