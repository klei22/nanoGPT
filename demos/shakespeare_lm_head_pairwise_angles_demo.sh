#!/bin/bash
# Train two tiny shakespeare_char models with different seeds, then compare
# lm_head pairwise vocabulary-vector angles and emit an interactive HTML report.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
OUT_ROOT="out/shakespeare_lm_head_pairwise_angles_demo"
RUN_A="${OUT_ROOT}/seed_1337"
RUN_B="${OUT_ROOT}/seed_2024"
REPORT_DIR="${OUT_ROOT}/analysis"

mkdir -p "${REPORT_DIR}"

pushd "${DATA_DIR}" > /dev/null
bash get_dataset.sh
if [[ ! -f train.bin || ! -f val.bin || ! -f meta.pkl ]]; then
  python3 prepare.py -t input.txt --method char
fi
popd > /dev/null

train_one() {
  local seed="$1"
  local out_dir="$2"
  rm -rf "${out_dir}"
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${out_dir}" \
    --init_from scratch \
    --seed "${seed}" \
    --max_iters 200 \
    --eval_interval 100 \
    --eval_iters 20 \
    --log_interval 20 \
    --always_save_checkpoint \
    --block_size 64 \
    --batch_size 32 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 128 \
    --learning_rate 1e-3 \
    --device cuda \
    --no-compile || \
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${out_dir}" \
    --init_from scratch \
    --seed "${seed}" \
    --max_iters 200 \
    --eval_interval 100 \
    --eval_iters 20 \
    --log_interval 20 \
    --always_save_checkpoint \
    --block_size 64 \
    --batch_size 32 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 128 \
    --learning_rate 1e-3 \
    --device cpu \
    --no-compile
}

train_one 1337 "${RUN_A}"
train_one 2024 "${RUN_B}"

python3 analysis/lm_head_pairwise_angles/compare_lm_head_pairwise_angles.py \
  "${RUN_A}/ckpt.pt" "${RUN_B}/ckpt.pt" \
  --meta "${DATA_DIR}/meta.pkl" \
  --device auto \
  --min-angle 0 --max-angle 180 \
  --csv "${REPORT_DIR}/lm_head_pairwise_pairs.csv" \
  --html "${REPORT_DIR}/lm_head_pairwise_report.html"

cat <<MSG
Wrote ${REPORT_DIR}/lm_head_pairwise_report.html
To launch the selector webapp for these and other checkpoints:
  python3 analysis/lm_head_pairwise_angles/app.py --ckpt-root ${OUT_ROOT}
MSG
