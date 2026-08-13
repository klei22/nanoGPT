#!/usr/bin/env bash
# Train a width-3 model and export every saved embedding snapshot for Three.js.
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
MAX_ITERS="${MAX_ITERS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
OUT_DIR="${OUT_DIR:-out/digits_3d}"
DATA_DIR="data/digits_3d"
VIEW_DIR="report/threejs/digits-3d"
WTE_FIXED_NORM="${WTE_FIXED_NORM:-true}"
WTE_FIXED_NORM_VALUE="${WTE_FIXED_NORM_VALUE:-}"
WTE_WEIGHT_TYING="${WTE_WEIGHT_TYING:-true}"
NUM_DIGITS="${NUM_DIGITS:-10}"
NUM_LETTERS="${NUM_LETTERS:-10}"
EMBEDDING_DIM="${EMBEDDING_DIM:-3}"
TRAJECTORY_FILE="${TRAJECTORY_FILE:-${VIEW_DIR}/token_trajectories.json}"
DROPOUT_PERCENT="${DROPOUT_PERCENT:-}"
DROPOUT_COUNT="${DROPOUT_COUNT:-1}"

case "${WTE_FIXED_NORM}" in
  true|1|yes) WTE_NORM_ARGS=(--wte_fixed_norm) ;;
  false|0|no) WTE_NORM_ARGS=(--no-wte_fixed_norm) ;;
  *) echo "WTE_FIXED_NORM must be true or false" >&2; exit 2 ;;
esac
if [ -n "${WTE_FIXED_NORM_VALUE}" ]; then
  WTE_NORM_ARGS+=(--wte_fixed_norm_value "${WTE_FIXED_NORM_VALUE}")
fi
case "${WTE_WEIGHT_TYING}" in
  true|1|yes) WTE_TYING_ARGS=(--wte_weight_tying) ;;
  false|0|no) WTE_TYING_ARGS=(--no-wte_weight_tying) ;;
  *) echo "WTE_WEIGHT_TYING must be true or false" >&2; exit 2 ;;
esac

python3 "${DATA_DIR}/prepare.py" --num-digits "${NUM_DIGITS}" --num-letters "${NUM_LETTERS}"

TRAIN_ARGS=(
  --dataset digits_3d \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --block_size 10 \
  --batch_size 64 \
  --n_layer 1 \
  --n_head 1 \
  --n_embd "${EMBEDDING_DIM}" \
  "${WTE_NORM_ARGS[@]}" \
  "${WTE_TYING_ARGS[@]}" \
  --dropout 0.0 \
  --eval_interval "${SAVE_INTERVAL}" \
  --eval_iters 20 \
  --save_major_ckpt_interval "${SAVE_INTERVAL}" \
  --always_save_checkpoint \
  --learning_rate 3e-3 \
  --min_lr 3e-4 \
  --warmup_iters 20 \
  --decay_lr \
  --no-compile
)

if [ -n "${DROPOUT_PERCENT}" ]; then
  if [ "${DROPOUT_PERCENT}" -le 0 ] || [ "${DROPOUT_PERCENT}" -ge 100 ]; then
    echo "DROPOUT_PERCENT must be between 1 and 99" >&2; exit 2
  fi
  DROPOUT_ITERATION=$((MAX_ITERS * DROPOUT_PERCENT / 100))
  python3 train.py "${TRAIN_ARGS[@]}" --max_iters "${DROPOUT_ITERATION}"
  python3 "${DATA_DIR}/prepare.py" --num-digits "${NUM_DIGITS}" --num-letters "${NUM_LETTERS}" --dropout-count "${DROPOUT_COUNT}"
  python3 train.py "${TRAIN_ARGS[@]}" --max_iters "${MAX_ITERS}" --init_from resume
  DROPOUT_EXPORT_ARGS=(--dropout-iteration "${DROPOUT_ITERATION}")
else
  python3 train.py "${TRAIN_ARGS[@]}" --max_iters "${MAX_ITERS}"
  DROPOUT_EXPORT_ARGS=()
fi

python3 analysis/export_3d_token_trajectories.py \
  --checkpoint-dir "${OUT_DIR}" \
  --meta "${DATA_DIR}/meta.pkl" \
  --output "${TRAJECTORY_FILE}" \
  "${DROPOUT_EXPORT_ARGS[@]}"

cat <<EOF
Done. Serve the repository (fetch does not work from file://), then open:
  python3 -m http.server 8000
  http://localhost:8000/${VIEW_DIR}/index.html?data=${TRAJECTORY_FILE#${VIEW_DIR}/}
The ${NUM_DIGITS} digit-like symbols are trained; ${NUM_LETTERS} letters are vocabulary-only controls.
Embedding dimension: ${EMBEDDING_DIM} (dimensions above 3 are globally PCA-projected for viewing).
WTE/LM-head weight tying: ${WTE_WEIGHT_TYING}.
EOF
