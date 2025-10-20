#!/bin/bash
# executorch_android_export_demo.sh
#
# Run a lightweight end-to-end workflow that trains a tiny nanoGPT model,
# exports it to ExecuTorch format, copies the artifacts into the Android app,
# and demonstrates how to push them to a device via ADB.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
OUT_DIR="out/executorch_android_demo"
EXPORT_DIR="${OUT_DIR}/executorch_artifacts"
MODEL_NAME="${MODEL_NAME:-shakespeare-char-demo}"
BLOCK_SIZE=${BLOCK_SIZE:-256}
TRAIN_ITERS=${TRAIN_ITERS:-200}
BATCH_SIZE=${BATCH_SIZE:-64}
ANDROID_APP_ROOT="${ANDROID_APP_ROOT:-EdgeAIApp-ExecuTorch}"
ANDROID_PUSH_DRY_RUN="${ANDROID_PUSH_DRY_RUN:-1}"
ANDROID_DEVICE_DIR_OVERRIDE="${ANDROID_DEVICE_DIR_OVERRIDE:-}"
ANDROID_ADB_BIN="${ANDROID_ADB_BIN:-}"

CKPT_PATH="${OUT_DIR}/ckpt.pt"
META_PATH="${DATA_DIR}/meta.pkl"
PTE_PATH="${EXPORT_DIR}/${MODEL_NAME}.pte"

mkdir -p "${OUT_DIR}"

step() {
  echo
  echo "=== $1 ==="
}

step "Step 1: Prepare the Shakespeare character dataset"
if [ ! -f "${DATA_DIR}/train.bin" ] || [ ! -f "${DATA_DIR}/val.bin" ] || [ ! -f "${META_PATH}" ]; then
  bash "${DATA_DIR}/get_dataset.sh"
else
  echo "Found existing tokenized dataset artifacts in ${DATA_DIR}."
fi

step "Step 2: Train a compact nanoGPT model"
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${OUT_DIR}" \
  --block_size "${BLOCK_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --max_iters "${TRAIN_ITERS}" \
  --lr_decay_iters "${TRAIN_ITERS}" \
  --eval_interval 50 \
  --eval_iters 50 \
  --log_interval 10 \
  --learning_rate 3e-4 \
  --dropout 0.0 \
  --always_save_checkpoint

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

step "Step 3: Export the checkpoint to ExecuTorch (.pte)"
python3 export_model/executorch/export_to_executorch.py \
  --checkpoint "${CKPT_PATH}" \
  --output-dir "${EXPORT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --block-size "${BLOCK_SIZE}" \
  --extra-asset "${META_PATH}" \
  --verbose

if [ ! -f "${PTE_PATH}" ]; then
  echo "ExecuTorch program not found at ${PTE_PATH}" >&2
  exit 1
fi

step "Step 4: Copy artifacts into the Android demo app"
COPY_ARGS=(
  --artifacts "${EXPORT_DIR}"
  --app "${ANDROID_APP_ROOT}"
  --model-name "${MODEL_NAME}"
  --clear-target
  --install-default-name
  --overwrite-default
  --verbose
)
python3 export_model/executorch/copy_assets.py "${COPY_ARGS[@]}"

step "Step 5: Push artifacts to a connected Android device"
PUSH_ARGS=(--artifacts "${EXPORT_DIR}" --verbose)
if [ -n "${ANDROID_DEVICE_DIR_OVERRIDE}" ]; then
  PUSH_ARGS+=(--device-dir "${ANDROID_DEVICE_DIR_OVERRIDE}")
fi
if [ -n "${ANDROID_ADB_BIN}" ]; then
  PUSH_ARGS+=(--adb "${ANDROID_ADB_BIN}")
fi
if [ "${ANDROID_PUSH_DRY_RUN}" != "0" ]; then
  PUSH_ARGS+=(--dry-run)
  echo "(Dry run enabled. Set ANDROID_PUSH_DRY_RUN=0 to execute ADB commands.)"
fi
python3 export_model/executorch/push_to_android.py "${PUSH_ARGS[@]}"

echo
cat <<MSG
Workflow complete!
- Training checkpoint: ${CKPT_PATH}
- ExecuTorch artifacts: ${EXPORT_DIR}
- Android assets folder updated under: ${ANDROID_APP_ROOT}
MSG
