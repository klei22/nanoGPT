#!/bin/bash
# sparse_autoencoder_demo.sh
# End-to-end walkthrough for training a small GPT checkpoint and fitting
# a sparse autoencoder on its activations.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
OUT_DIR="out/shakespeare_sae_demo"
AE_DIR="${OUT_DIR}/autoencoder"
CKPT_PATH="${OUT_DIR}/ckpt.pt"

mkdir -p "${DATA_DIR}" "${OUT_DIR}" "${AE_DIR}"

echo "=== Step 1: Prepare the Shakespeare character dataset ==="
pushd "${DATA_DIR}" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

echo "=== Step 2: Train a lightweight GPT checkpoint ==="
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${OUT_DIR}" \
  --block_size 256 \
  --batch_size 48 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --max_iters 200 \
  --eval_interval 50 \
  --eval_iters 20 \
  --learning_rate 3e-4 \
  --compile

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

echo "=== Step 3: Fit a sparse autoencoder to the final block activations ==="
python3 train_sparse_autoencoder.py \
  --checkpoint "${CKPT_PATH}" \
  --dataset shakespeare_char \
  --out_dir "${AE_DIR}" \
  --block_size 256 \
  --batch_size 8 \
  --layer -1 \
  --activation_source block_output \
  --latent_dim 128 \
  --train_steps 200 \
  --eval_interval 20 \
  --save_interval 100

echo "Sparse autoencoder demo complete. Check ${AE_DIR} for checkpoints and logs."
