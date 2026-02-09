#!/bin/bash
# demos/orthomerge_minipile_opus100_demo.sh
#
# Demo pipeline:
#   1) Prepare + train 10k iterations on minipile (tiktoken).
#   2) Download + tiktoken-tokenize OPUS-100 en-pt and en-id with -s/-S.
#   3) Finetune each for 2k iterations from the minipile checkpoint.
#   4) Merge the two finetuned checkpoints with l2/simple/orthomerge and sample.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
RESET='\033[0m'

MINIPILE_DIR="data/minipile"
OPUS_ROOT="data/opus-100"
OPUS_EN_PT_DIR="${OPUS_ROOT}/en-pt"
OPUS_EN_ID_DIR="${OPUS_ROOT}/en-id"

TOKEN_SUBDIR_EN_PT="tiktoken_en_pt"
TOKEN_SUBDIR_EN_ID="tiktoken_en_id"

BASE_OUT_DIR="out_minipile_10k"
FT_EN_PT_OUT_DIR="out_opus_en_pt_2k"
FT_EN_ID_OUT_DIR="out_opus_en_id_2k"

MERGE_L2_OUT_DIR="out_merge_l2"
MERGE_SIMPLE_OUT_DIR="out_merge_simple"
MERGE_ORTHO_OUT_DIR="out_merge_orthomerge"

BLOCK_SIZE=256
BATCH_SIZE=64
EVAL_ITERS=200

prepare_minipile() {
  if [ ! -d "${MINIPILE_DIR}" ]; then
    echo -e "${RED}[ERROR]${RESET} Missing dataset directory: ${MINIPILE_DIR}" >&2
    exit 1
  fi

  echo -e "${CYAN}=== Preparing minipile dataset ===${RESET}"
  pushd "${MINIPILE_DIR}" > /dev/null

  if [ ! -f "input.txt" ]; then
    echo -e "${MAGENTA}[OBTAIN]${RESET} Downloading minipile"
    bash get_dataset.sh
  else
    echo -e "${YELLOW}[SKIP]${RESET} Found input.txt for minipile"
  fi

  if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
    echo -e "${GREEN}[TOKENIZE]${RESET} Tokenizing minipile with tiktoken"
    python3 prepare.py -t input.txt --method tiktoken
  else
    echo -e "${YELLOW}[SKIP]${RESET} Found tokenized minipile artifacts"
  fi

  popd > /dev/null
}

prepare_opus_pair() {
  local pair_dir="$1"
  local to_lang="$2"
  local token_suffix="$3"
  local token_subdir="tiktoken_${token_suffix}"

  echo -e "${CYAN}=== Preparing OPUS-100 en-${to_lang} dataset ===${RESET}"
  mkdir -p "${pair_dir}"
  pushd "${pair_dir}" > /dev/null

  if [ ! -f "input.txt" ]; then
    echo -e "${MAGENTA}[OBTAIN]${RESET} Downloading OPUS-100 en-${to_lang}"
    python3 ../get_dataset.py -f en -t "${to_lang}" -o input.txt
  else
    echo -e "${YELLOW}[SKIP]${RESET} Found input.txt for en-${to_lang}"
  fi

  if [ ! -f "${token_subdir}/train.bin" ] || [ ! -f "${token_subdir}/val.bin" ] || [ ! -f "${token_subdir}/meta.pkl" ]; then
    echo -e "${GREEN}[TOKENIZE]${RESET} Tokenizing en-${to_lang} with tiktoken (-s/-S)"
    python3 ../prepare.py -t input.txt --method tiktoken -s -S "${token_suffix}"
  else
    echo -e "${YELLOW}[SKIP]${RESET} Found tokenized en-${to_lang} artifacts"
  fi

  popd > /dev/null
}

train_minipile_base() {
  echo -e "${CYAN}=== Training minipile base model (10k iters) ===${RESET}"
  if [ ! -f "${BASE_OUT_DIR}/ckpt.pt" ]; then
    python3 train.py \
      --dataset minipile \
      --out_dir "${BASE_OUT_DIR}" \
      --n_layer 6 \
      --n_head 6 \
      --n_embd 384 \
      --use_rotary_embeddings \
      --no-use_abs_pos_embeddings \
      --use_qk_norm \
      --use_qk_norm_scale \
      --use_peri_ln \
      --block_size "${BLOCK_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --max_iters 10000 \
      --eval_interval 10000 \
      --eval_iters "${EVAL_ITERS}" \
      --eta_variant "iteration" \
      --compile
  else
    echo -e "${YELLOW}[SKIP]${RESET} Found base checkpoint at ${BASE_OUT_DIR}/ckpt.pt"
  fi
}

finetune_opus() {
  local dataset_path="$1"
  local out_dir="$2"
  local label="$3"

  echo -e "${CYAN}=== Finetuning ${label} (2k iters) ===${RESET}"
  if [ ! -f "${out_dir}/ckpt.pt" ]; then
    python3 train.py \
      --dataset "${dataset_path}" \
      --out_dir "${out_dir}" \
      --init_from prev_run \
      --prev_run_ckpt "${BASE_OUT_DIR}" \
      --max_iters 2000 \
      --eval_interval 2000 \
      --eval_iters "${EVAL_ITERS}" \
      --eta_variant "iteration" \
      --compile
  else
    echo -e "${YELLOW}[SKIP]${RESET} Found finetuned checkpoint at ${out_dir}/ckpt.pt"
  fi
}

merge_models() {
  local merge_mode="$1"
  local out_dir="$2"

  echo -e "${CYAN}=== Merging checkpoints (${merge_mode}) ===${RESET}"
  if [ ! -f "${out_dir}/ckpt.pt" ]; then
    if [ "${merge_mode}" = "orthomerge" ]; then
      python3 model_merge.py "${FT_EN_PT_OUT_DIR}" "${FT_EN_ID_OUT_DIR}" \
        --out_dir "${out_dir}" \
        --merge_mode orthomerge \
        --base_ckpt_dir "${BASE_OUT_DIR}"
    else
      python3 model_merge.py "${FT_EN_PT_OUT_DIR}" "${FT_EN_ID_OUT_DIR}" \
        --out_dir "${out_dir}" \
        --merge_mode "${merge_mode}"
    fi
  else
    echo -e "${YELLOW}[SKIP]${RESET} Found merged checkpoint at ${out_dir}/ckpt.pt"
  fi
}

sample_merge() {
  local out_dir="$1"
  local label="$2"
  local sample_file="${out_dir}/sample_output.txt"

  echo -e "${CYAN}=== Sampling from ${label} ===${RESET}"
  python3 sample.py \
    --out_dir "${out_dir}" \
    --start "English: The weather is nice today.\nTranslation:" \
    --num_samples 1 \
    --max_new_tokens 200 \
    --sample_file "${sample_file}"

  echo -e "${GREEN}--- Sample output (${label}) ---${RESET}"
  cat "${sample_file}"
  echo
}

prepare_minipile
prepare_opus_pair "${OPUS_EN_PT_DIR}" "pt" "en_pt"
prepare_opus_pair "${OPUS_EN_ID_DIR}" "id" "en_id"

train_minipile_base

finetune_opus "opus-100/en-pt/${TOKEN_SUBDIR_EN_PT}" "${FT_EN_PT_OUT_DIR}" "OPUS en-pt"
finetune_opus "opus-100/en-id/${TOKEN_SUBDIR_EN_ID}" "${FT_EN_ID_OUT_DIR}" "OPUS en-id"

merge_models "l2" "${MERGE_L2_OUT_DIR}"
merge_models "simple" "${MERGE_SIMPLE_OUT_DIR}"
merge_models "orthomerge" "${MERGE_ORTHO_OUT_DIR}"

sample_merge "${MERGE_L2_OUT_DIR}" "L2 merge"
sample_merge "${MERGE_SIMPLE_OUT_DIR}" "Simple merge"
sample_merge "${MERGE_ORTHO_OUT_DIR}" "OrthoMerge"
