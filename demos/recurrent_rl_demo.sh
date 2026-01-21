#!/bin/bash
# demos/recurrent_rl_demo.sh
# Demonstrates recurrent training with an RL reward interval using the
# lightweight SimpleSnake gym environment.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
OUT_DIR="out/recurrent_rl_demo"
CKPT_PATH="${OUT_DIR}/ckpt.pt"
RL_OUT_DIR="${OUT_DIR}/recurrent_rl"

mkdir -p "${DATA_DIR}"

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd "${DATA_DIR}" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "${OUT_DIR}"

echo "=== Step 2: Train a tiny baseline checkpoint for recurrent fine-tuning ==="
if [ ! -f "${CKPT_PATH}" ]; then
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${OUT_DIR}" \
    --batch_size 12 \
    --block_size 128 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 128 \
    --max_iters 80 \
    --eval_interval 40 \
    --eval_iters 40 \
    --learning_rate 5e-4 \
    --weight_decay 0.1
else
  echo "Found existing checkpoint at ${CKPT_PATH}."
fi

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

mkdir -p "${RL_OUT_DIR}"

echo "=== Step 3: Run recurrent RL fine-tuning with SimpleSnake rewards ==="
python3 train_recurrent.py \
  --resume_ckpt "${CKPT_PATH}" \
  --dataset shakespeare_char \
  --out_dir "${RL_OUT_DIR}" \
  --block_size 128 \
  --latent_steps 32 \
  --skip_steps 8 \
  --max_iters 50 \
  --eval_interval 25 \
  --eval_iters 10 \
  --rl_game simple_snake \
  --rl_interval 4 \
  --rl_weight 0.2 \
  --rl_action_dim 3 \
  --reset_optim

cat <<MSG
Recurrent RL demo complete.
Baseline checkpoint: ${CKPT_PATH}
Recurrent outputs:   ${RL_OUT_DIR}
MSG
