#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-out_mc_shakespeare_articulatory}"
SAMPLE_DIR="${SAMPLE_DIR:-${OUT_DIR}/samples}"
MAX_ITERS="${MAX_ITERS:-2000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# Build aligned lanes: lowercase, case, and articulatory-phonetic annotations.
data/shakespeare_char_articulatory/get_dataset.sh

DATASETS=(
  shakespeare_char_articulatory/lowercase
  shakespeare_char_articulatory/case
  shakespeare_char_articulatory/phonetic_class
  shakespeare_char_articulatory/place
  shakespeare_char_articulatory/manner
  shakespeare_char_articulatory/voicing
  shakespeare_char_articulatory/vowel_height
  shakespeare_char_articulatory/vowel_backness
  shakespeare_char_articulatory/vowel_rounding
)

python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --max_iters "${MAX_ITERS}" \
  --dropout 0.2 \
  --top_k 1 \
  --sample_each_eval \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --out_dir "${OUT_DIR}" \
  --compile

mkdir -p "${SAMPLE_DIR}"
RAW_SAMPLE="${SAMPLE_DIR}/sample_raw.txt"
python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --multicontext_start "but " "u___" "CVCC" "VAVA" "VSFS" "Vxvx" "oCC_" "fCC_" "uCC_" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --top_k 1 \
  --num_samples 1 \
  --sample_file "${RAW_SAMPLE}" | tee "${SAMPLE_DIR}/sample_stdout.txt"

python3 data/shakespeare_char_articulatory/recombine_case_sample.py \
  "${RAW_SAMPLE}" \
  --out_dir "${SAMPLE_DIR}" | tee "${SAMPLE_DIR}/recombined_stdout.txt"
