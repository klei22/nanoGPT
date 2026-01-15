#!/bin/bash
# offline_distillation_shakespeare_char_demo.sh
# Demonstrates offline knowledge distillation on the shakespeare_char dataset.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
TEACHER_OUT="out/shakespeare_char_teacher"
STUDENT_OUT="out/shakespeare_char_student_offline"
TEACHER_CKPT="${TEACHER_OUT}/ckpt.pt"
LOGITS_DIR="${TEACHER_OUT}/offline_logits"
TRAIN_LOGITS="${LOGITS_DIR}/train_logits.npy"
VAL_LOGITS="${LOGITS_DIR}/val_logits.npy"

mkdir -p "${DATA_DIR}"

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd "${DATA_DIR}" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

echo "=== Step 2: Train a small teacher model ==="
mkdir -p "${TEACHER_OUT}"
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${TEACHER_OUT}" \
  --block_size 64 \
  --batch_size 32 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --max_iters 200 \
  --eval_interval 50 \
  --eval_iters 50 \
  --learning_rate 1e-3

if [ ! -f "${TEACHER_CKPT}" ]; then
  echo "Expected teacher checkpoint not found at ${TEACHER_CKPT}" >&2
  exit 1
fi

echo "=== Step 3: Export offline logits from the teacher ==="
mkdir -p "${LOGITS_DIR}"
python3 demos/offline_distillation_export_logits.py \
  --ckpt_path "${TEACHER_CKPT}" \
  --dataset shakespeare_char \
  --split train \
  --output "${TRAIN_LOGITS}" \
  --batch_size 16

python3 demos/offline_distillation_export_logits.py \
  --ckpt_path "${TEACHER_CKPT}" \
  --dataset shakespeare_char \
  --split val \
  --output "${VAL_LOGITS}" \
  --batch_size 16

echo "=== Step 4: Train a student model using offline distillation ==="
mkdir -p "${STUDENT_OUT}"
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${STUDENT_OUT}" \
  --block_size 64 \
  --batch_size 32 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 128 \
  --max_iters 200 \
  --eval_interval 50 \
  --eval_iters 50 \
  --learning_rate 1e-3 \
  --distillation_loss kl_divergence \
  --distillation_weight 1.0 \
  --distillation_teacher_logits_train "${TRAIN_LOGITS}" \
  --distillation_teacher_logits_val "${VAL_LOGITS}"

cat <<MSG
Offline distillation demo complete.
Teacher outputs: ${TEACHER_OUT}
Student outputs: ${STUDENT_OUT}
MSG
