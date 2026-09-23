#!/usr/bin/env bash
set -euo pipefail

# End-to-end demo for the sequential latent-mix exploration:
#   1. Ensure minipile and dialogsum have input.txt, train.bin, and val.bin.
#   2. Run explorations/sequential_minipile_dialogsum_recurrent_latent_mix.yaml.
#   3. Sample from the final recurrent checkpoint for each latent_mix_mode.
#
# Useful overrides:
#   AUTO_PREPARE_DATA=0  # only check data and print instructions if files are missing
#   RUN_TRAINING=0       # skip training and sample from existing checkpoints
#   OUTPUT_DIR=out       # run_experiments output root
#   DEVICE=cuda:0        # training/sampling device used by the YAML and sample.py
#   MAX_NEW_TOKENS=200   # generated tokens per sample
#   NUM_SAMPLES=1        # samples per checkpoint

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-explorations/sequential_minipile_dialogsum_recurrent_latent_mix.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-out}"
DEVICE="${DEVICE:-cuda:0}"
AUTO_PREPARE_DATA="${AUTO_PREPARE_DATA:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
START_TEXT="${START_TEXT:-#U: Please summarize the following:\nA researcher tested recurrent latent mixing after pretraining on a broad corpus and finetuning on summaries.\n#B:\n}"

require_file_or_prepare() {
  local dataset="$1"
  local dataset_dir="data/${dataset}"
  local input_path="${dataset_dir}/input.txt"
  local train_path="${dataset_dir}/train.bin"
  local val_path="${dataset_dir}/val.bin"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "[ERROR] Missing dataset directory: ${dataset_dir}" >&2
    return 1
  fi

  if [[ ! -f "${input_path}" ]]; then
    echo "[WARN] Missing ${input_path}."
    if [[ "${AUTO_PREPARE_DATA}" != "1" ]]; then
      cat <<MSG
Set AUTO_PREPARE_DATA=1 or run manually:
  (cd ${dataset_dir} && bash get_dataset.sh)
  (cd ${dataset_dir} && python3 prepare.py -t input.txt --method tiktoken --train_output train.bin --val_output val.bin)
MSG
      return 1
    fi
    echo "[INFO] Downloading/building ${input_path} via ${dataset_dir}/get_dataset.sh ..."
    (cd "${dataset_dir}" && bash get_dataset.sh)
  fi

  if [[ ! -f "${train_path}" || ! -f "${val_path}" ]]; then
    echo "[WARN] Missing ${train_path} and/or ${val_path}."
    if [[ "${AUTO_PREPARE_DATA}" != "1" ]]; then
      cat <<MSG
Set AUTO_PREPARE_DATA=1 or run manually:
  (cd ${dataset_dir} && python3 prepare.py -t input.txt --method tiktoken --train_output train.bin --val_output val.bin)
MSG
      return 1
    fi
    echo "[INFO] Tokenizing ${dataset}/input.txt with tiktoken into train.bin and val.bin ..."
    (cd "${dataset_dir}" && python3 prepare.py -t input.txt --method tiktoken --train_output train.bin --val_output val.bin)
  fi

  echo "[OK] ${dataset}: found input.txt, train.bin, and val.bin."
}

ensure_data() {
  require_file_or_prepare "minipile"
  require_file_or_prepare "dialogsum"
}

run_training_sequence() {
  echo "[INFO] Running sequential exploration config: ${CONFIG}"
  python3 optimization_and_search/run_experiments.py \
    --config "${CONFIG}" \
    --config_format yaml \
    --output_dir "${OUTPUT_DIR}"
}

sample_variant() {
  local variant="$1"
  local run_name="seq_minipile_dialogsum_recurrent_${variant}"
  local recurrent_stage="03_recurrent_dialogsum_${variant}"
  local ckpt_dir="${OUTPUT_DIR}/${run_name}/${recurrent_stage}"
  local ckpt_path="${ckpt_dir}/ckpt.pt"
  local sample_dir="${OUTPUT_DIR}/${run_name}/samples"
  local sample_file="${sample_dir}/${variant}_sample.txt"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[ERROR] Expected final checkpoint not found: ${ckpt_path}" >&2
    echo "        Run this script with RUN_TRAINING=1, or inspect ${OUTPUT_DIR}/${run_name}." >&2
    return 1
  fi

  mkdir -p "${sample_dir}"
  echo "[INFO] Sampling ${variant} recurrent checkpoint: ${ckpt_path}"
  python3 sample.py \
    --init_from resume \
    --out_dir "${ckpt_dir}" \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --start "${START_TEXT}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --top_k 200 \
    --sample_file "${sample_file}" \
    --no-print_model_info
  echo "[OK] Wrote ${sample_file}"
}

main() {
  ensure_data

  if [[ "${RUN_TRAINING}" == "1" ]]; then
    run_training_sequence
  else
    echo "[INFO] RUN_TRAINING=0; skipping sequential training and using existing checkpoints."
  fi

  sample_variant "direct"
  sample_variant "slerp"
  sample_variant "add_norm"

  echo "[DONE] Sequential recurrent latent-mix demo completed."
}

main "$@"
