#!/bin/bash
# demos/shakespeare_ptq_comparison_demo.sh
#
# End-to-end demonstration that trains a compact model on the shakespeare_char
# dataset, applies post-training quantization across multiple bit-widths, and
# compares each quantized checkpoint against the floating point baseline using
# the checkpoint regex explorer's L2 norm and angle/cosine statistics.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
OUT_ROOT="out/shakespeare_ptq_comparison_demo"
BASE_OUT="${OUT_ROOT}/fp32"
QUANT_ROOT="${OUT_ROOT}/quantized"
HIST_ROOT="${OUT_ROOT}/histograms"
CSV_ROOT="${OUT_ROOT}/comparison_csv"
PATTERN='transformer\\.h\\.[0-9]+\\.(attn\\.(c_attn_(q|k|v)|c_proj)|mlp\\.(c_fc|c_proj))\\.weight'
BITS=(8 6 4)

mkdir -p "${DATA_DIR}"

cat <<'MSG'
=== Step 1: Prepare the shakespeare_char dataset ===
MSG
pushd "${DATA_DIR}" > /dev/null
bash get_dataset.sh
popd > /dev/null

cat <<'MSG'
=== Step 2: Train the floating point baseline model ===
MSG
rm -rf "${OUT_ROOT}"
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${BASE_OUT}" \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --block_size 256 \
  --batch_size 64 \
  --max_iters 750 \
  --eval_interval 150 \
  --eval_iters 200 \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --compile

if [ ! -f "${BASE_OUT}/ckpt.pt" ]; then
  echo "Floating point checkpoint missing at ${BASE_OUT}/ckpt.pt" >&2
  exit 1
fi

mkdir -p "${QUANT_ROOT}" "${HIST_ROOT}" "${CSV_ROOT}"

cat <<'MSG'
=== Step 3: Evaluate the floating point baseline ===
MSG
python3 sample.py \
  --out_dir "${BASE_OUT}" \
  --eval_only \
  --eval_dataset shakespeare_char

step=4
for bit in "${BITS[@]}"; do
  QUANT_DIR="${QUANT_ROOT}/${bit}bit"
  HIST_DIR="${HIST_ROOT}/${bit}bit"
  CSV_PATH="${CSV_ROOT}/ptq_${bit}bit_vs_fp32.csv"

  echo "=== Step ${step}: Quantize to ${bit}-bit weights ==="
  ((step++))
  rm -rf "${QUANT_DIR}" "${HIST_DIR}"
  rm -f "${CSV_PATH}"
  python3 quantizations/ptq/fake_quantize_ckpt.py "${BASE_OUT}" \
    --out_dir "${QUANT_DIR}" \
    --num_bits "${bit}"

  echo "=== Step ${step}: Evaluate the ${bit}-bit checkpoint ==="
  ((step++))
  python3 sample.py \
    --out_dir "${QUANT_DIR}" \
    --eval_only \
    --eval_dataset shakespeare_char

  echo "=== Step ${step}: Compare ${bit}-bit checkpoint with fp32 baseline ==="
  ((step++))
  python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
    "${QUANT_DIR}/ckpt.pt" \
    "${PATTERN}" \
    --compare-ckpt "${BASE_OUT}/ckpt.pt" \
    --histogram-dir "${HIST_DIR}" \
    --histogram-bins 60 \
    --comparison-csv "${CSV_PATH}" \
    --max-rows 20 \
    --max-l2-rows 20 \
    --max-comparison-rows 20

done

cat <<EOF
Summary artifacts written under ${OUT_ROOT}:
  - Floating point baseline checkpoint: ${BASE_OUT}/ckpt.pt
  - Quantized checkpoints: ${QUANT_ROOT}/<bit>bit/ckpt.pt
  - L2 norm and comparison histograms: ${HIST_ROOT}/<bit>bit/
  - Angle/cosine summary CSV exports: ${CSV_ROOT}/ptq_<bit>bit_vs_fp32.csv
EOF
