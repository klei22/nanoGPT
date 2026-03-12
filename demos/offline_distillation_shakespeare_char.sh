#!/bin/bash
# demos/offline_distillation_shakespeare_char.sh
# Demonstrates offline distillation on the shakespeare_char dataset using
# precomputed teacher logits (block_size=1 for exact alignment).

set -euo pipefail

DATA_DIR="data/shakespeare_char"
TEACHER_OUT="out/offline_distill_teacher_shakespeare_char"
STUDENT_OUT="out/offline_distill_student_shakespeare_char"
LOGITS_DIR="out/offline_distill_logits_shakespeare_char"
TRAIN_LOGITS="${LOGITS_DIR}/teacher_logits_train.npy"
VAL_LOGITS="${LOGITS_DIR}/teacher_logits_val.npy"
TEACHER_CKPT="${TEACHER_OUT}/ckpt.pt"

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd "${DATA_DIR}" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "${TEACHER_OUT}" "${LOGITS_DIR}"

if [ ! -f "${TEACHER_CKPT}" ]; then
  echo "=== Step 2: Train a compact teacher model (block_size=1) ==="
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${TEACHER_OUT}" \
    --block_size 1 \
    --batch_size 128 \
    --n_layer 2 \
    --n_head 2 \
    --n_embd 64 \
    --max_iters 200 \
    --eval_interval 50 \
    --eval_iters 20 \
    --learning_rate 3e-4 \
    --no-tensorboard_log
else
  echo "Found existing teacher checkpoint at ${TEACHER_CKPT}."
fi

echo "=== Step 3: Generate offline logits from the teacher ==="
CKPT_PATH="${TEACHER_CKPT}" \
TRAIN_LOGITS="${TRAIN_LOGITS}" \
VAL_LOGITS="${VAL_LOGITS}" \
DATA_DIR="${DATA_DIR}" \
python3 - <<'PY'
import os
import pickle

import numpy as np
import torch

from model import GPT, GPTConfig

torch.set_grad_enabled(False)

ckpt_path = os.environ["CKPT_PATH"]
data_dir = os.environ["DATA_DIR"]
train_logits_path = os.environ["TRAIN_LOGITS"]
val_logits_path = os.environ["VAL_LOGITS"]

checkpoint = torch.load(ckpt_path, map_location="cpu")
model_args = checkpoint["model_args"]
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

state_dict = checkpoint["model"]
for key in list(state_dict.keys()):
    if key.startswith("_orig_mod."):
        state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)
model.load_state_dict(state_dict)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

meta_path = os.path.join(data_dir, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
vocab_size = meta.get("vocab_size", gptconf.vocab_size)
dtype = np.uint32 if vocab_size == 100277 else np.uint16

def load_data(split):
    path = os.path.join(data_dir, f"{split}.bin")
    return np.memmap(path, dtype=dtype, mode="r")

def write_logits(data, out_path):
    num_tokens = len(data)
    logits = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_tokens, gptconf.vocab_size),
    )
    batch_size = 4096
    for start in range(0, num_tokens, batch_size):
        tokens = data[start : start + batch_size].astype(np.int64)
        x = torch.from_numpy(tokens).unsqueeze(1).to(device)
        out, _ = model(x, targets=None, iter_num=0, dataset_idx=None, loss_fn=None)
        logits[start : start + batch_size] = out.squeeze(1).float().cpu().numpy()
    logits.flush()

train_data = load_data("train")
val_data = load_data("val")

write_logits(train_data, train_logits_path)
write_logits(val_data, val_logits_path)
print(f"Wrote logits to {train_logits_path} and {val_logits_path}")
PY

echo "=== Step 4: Train a student model with offline distillation ==="
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${STUDENT_OUT}" \
  --block_size 1 \
  --batch_size 128 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --max_iters 200 \
  --eval_interval 50 \
  --eval_iters 20 \
  --learning_rate 3e-4 \
  --no-tensorboard_log \
  --distillation_teacher_logits_train "${TRAIN_LOGITS}" \
  --distillation_teacher_logits_val "${VAL_LOGITS}" \
  --distillation_loss kl_divergence \
  --distillation_weight 1.0

cat <<MSG
Offline distillation demo complete.
Teacher: ${TEACHER_OUT}
Student: ${STUDENT_OUT}
Logits:  ${LOGITS_DIR}
MSG
