#!/bin/bash
# demos/shakespeare_char_ptq_comparison_demo.sh
#
# End-to-end demonstration that trains a reference model on the shakespeare_char
# dataset, produces several post-training quantized checkpoints, and compares
# their directional statistics against the original floating-point model using
# the checkpoint regex explorer's L2 norm comparison features.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
BASE_OUT_DIR="out/shakespeare_char_ptq_comparison"
FP32_OUT_DIR="${BASE_OUT_DIR}/fp32_model"
FP32_CKPT_PATH="${FP32_OUT_DIR}/ckpt.pt"
HISTOGRAM_DIR="${BASE_OUT_DIR}/histograms"
CSV_DIR="${BASE_OUT_DIR}/comparison_csv"
REPORT_DIR="${BASE_OUT_DIR}/reports"
REGEX_PATTERN="transformer\\.h\\.[0-9]+\\.(attn\\.(c_attn_(q|k|v)|c_proj)|mlp\\.(c_fc|c_proj))\\.weight"

QUANT_LABELS=("int8_symmetric" "int8_asymmetric" "int4_symmetric")
QUANT_BITS=(8 8 4)
QUANT_SCHEMES=("symmetric" "asymmetric" "symmetric")

mkdir -p "${DATA_DIR}" "${BASE_OUT_DIR}" "${HISTOGRAM_DIR}" "${CSV_DIR}" "${REPORT_DIR}"

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd "${DATA_DIR}" > /dev/null
bash get_dataset.sh
popd > /dev/null

echo "=== Step 2: Train a floating-point reference model on shakespeare_char ==="
rm -rf "${FP32_OUT_DIR}"
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${FP32_OUT_DIR}" \
  --block_size 128 \
  --batch_size 64 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --max_iters 2000 \
  --eval_interval 200 \
  --eval_iters 200 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --compile

if [[ ! -f "${FP32_CKPT_PATH}" ]]; then
  echo "Expected checkpoint not found at ${FP32_CKPT_PATH}" >&2
  exit 1
fi

echo "=== Step 3: Evaluate the floating-point checkpoint ==="
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${FP32_OUT_DIR}" \
  --eval_only \
  --compute_model_stats \
  --print_model_stats_table "${REPORT_DIR}/fp32_model_stats.csv"

python3 sample.py \
  --out_dir "${FP32_OUT_DIR}" \
  --eval_only \
  --eval_dataset shakespeare_char

python3 sample.py \
  --out_dir "${FP32_OUT_DIR}" \
  --num_samples 1 \
  --max_new_tokens 80 \
  --start "ROMEO:" \
  --sample_file "${REPORT_DIR}/fp32_sample.txt"

echo "=== Step 4: Baseline L2 norm histograms for floating-point tensors ==="
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${FP32_CKPT_PATH}" \
  "${REGEX_PATTERN}" \
  --histogram-dir "${HISTOGRAM_DIR}/fp32" \
  --histogram-bins 60 \
  --pairwise-limit 0 \
  --comparison-limit 0

for idx in "${!QUANT_LABELS[@]}"; do
  label="${QUANT_LABELS[$idx]}"
  bits="${QUANT_BITS[$idx]}"
  scheme="${QUANT_SCHEMES[$idx]}"
  quant_out_dir="${BASE_OUT_DIR}/${label}"
  quant_ckpt_path="${quant_out_dir}/ckpt.pt"

  echo "=== Step 5: Apply ${bits}-bit ${scheme} PTQ (${label}) ==="
  python3 quantizations/ptq/fake_quantize_ckpt.py "${FP32_OUT_DIR}" \
    --num_bits "${bits}" \
    --quantization "${scheme}" \
    --out_dir "${quant_out_dir}"

  if [[ ! -f "${quant_ckpt_path}" ]]; then
    echo "Expected quantized checkpoint not found at ${quant_ckpt_path}" >&2
    exit 1
  fi

  echo "=== Step 6: Evaluate quantized checkpoint (${label}) ==="
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${quant_out_dir}" \
    --eval_only \
    --compute_model_stats \
    --print_model_stats_table "${REPORT_DIR}/${label}_model_stats.csv"

  python3 sample.py \
    --out_dir "${quant_out_dir}" \
    --eval_only \
    --eval_dataset shakespeare_char

  python3 sample.py \
    --out_dir "${quant_out_dir}" \
    --num_samples 1 \
    --max_new_tokens 80 \
    --start "ROMEO:" \
    --sample_file "${REPORT_DIR}/${label}_sample.txt"

  echo "=== Step 7: Compare quantized checkpoint (${label}) with floating-point reference ==="
  python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
    "${FP32_CKPT_PATH}" \
    "${REGEX_PATTERN}" \
    --compare-ckpt "${quant_ckpt_path}" \
    --histogram-dir "${HISTOGRAM_DIR}/${label}" \
    --histogram-bins 60 \
    --comparison-csv "${CSV_DIR}/${label}_comparison.csv" \
    --max-comparison-rows 20 \
    --angle-units degrees

done

echo "\nAll artifacts saved under ${BASE_OUT_DIR}."
