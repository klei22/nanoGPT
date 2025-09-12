#!/bin/bash
# demos/ptq_demo.sh

# 1. Prepare minipile dataset
pushd data/minipile
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for minipile"
fi
popd

# 2. Train a small model
python3 train.py \
  --dataset minipile \
  --out_dir out_ptq_demo \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --block_size 64 \
  --max_iters 100 \
  --eval_iters 20 \
  --log_interval 20 \
  --device cpu

# 3. Sample from the trained model before quantization
python3 sample.py \
  --out_dir out_ptq_demo \
  --num_samples 1 \
  --max_new_tokens 40 \
  --start "Hello" \
  --device cpu

# 4. Apply post-training fake quantization
python3 quantizations/ptq/uniform_fake_quant.py out_ptq_demo --out_dir out_ptq_demo_ptq --num_bits 8

# 5. Sample from the quantized model
python3 sample.py \
  --out_dir out_ptq_demo_ptq \
  --num_samples 1 \
  --max_new_tokens 40 \
  --start "Hello" \
  --device cpu

