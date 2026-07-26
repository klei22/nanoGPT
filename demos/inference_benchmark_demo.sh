#!/usr/bin/env bash
# Demonstrate the inference benchmark configs for addition_digits and OPUS-100 translation.
# Run from the repository root.
set -euo pipefail

# Keep wandb quiet for quick demos
export WANDB_MODE=offline

run_addition_demo() {
  echo "[addition] Preparing toy dataset and running a quick benchmarked training step..."
  pushd data/addition_digits >/dev/null
  python prepare.py --num-samples 2000 --max-number 50 --train-ratio 0.9
  popd >/dev/null

  python train.py \
    --dataset addition_digits \
    --out_dir out/demo_addition_benchmarks \
    --batch_size 32 \
    --block_size 64 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 128 \
    --dropout 0.0 \
    --learning_rate 3e-3 \
    --max_iters 60 \
    --lr_decay_iters 60 \
    --eval_interval 30 \
    --eval_iters 10 \
    --log_interval 10 \
    --device cpu \
    --benchmark_config benchmarks/examples/addition.json \
    --benchmark_top_k 5 20 \
    --benchmark_max_new_tokens 3
}

run_translation_demo() {
  echo "[translation] Running inference-only benchmark with a GPT-2 checkpoint..."
  python train.py \
    --dataset shakespeare_char \
    --block_size 256 \
    --batch_size 8 \
    --init_from gpt2 \
    --device cpu \
    --sample_only \
    --benchmark_config benchmarks/examples/opus100_translation.json \
    --benchmark_max_new_tokens 48 \
    --benchmark_top_k 50
}

run_addition_demo
run_translation_demo
