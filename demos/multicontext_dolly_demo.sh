#!/bin/bash
# demos/multicontext_dolly_demo.sh

set -euo pipefail

DATA_DIR="data/databricks-dolly-15k"
TRANSFORMS=(cvp in_word_position part_of_speech since_newline newlines_mod)

if [ ! -d "${DATA_DIR}" ]; then
  echo "Missing dataset directory: ${DATA_DIR}" >&2
  exit 1
fi

if [ ! -f "${DATA_DIR}/input.txt" ]; then
  pushd "${DATA_DIR}" > /dev/null
  bash get_dataset.sh
  popd > /dev/null
fi

if [ ! -f "${DATA_DIR}/train.bin" ] || [ ! -f "${DATA_DIR}/val.bin" ] || [ ! -f "${DATA_DIR}/meta.pkl" ]; then
  pushd "${DATA_DIR}" > /dev/null
  python3 prepare.py --method char -t input.txt
  popd > /dev/null
fi

build_variant() {
  local method="$1"
  local variant_dir="${DATA_DIR}/mc_${method}"

  if [ -f "${variant_dir}/train.bin" ] && [ -f "${variant_dir}/val.bin" ] && [ -f "${variant_dir}/meta.pkl" ]; then
    return
  fi

  mkdir -p "${variant_dir}"
  cp "${DATA_DIR}/input.txt" "${variant_dir}/input.txt"

  pushd "${variant_dir}" > /dev/null
  python3 ../utils/char_convert.py input.txt --method "${method}"
  python3 ../prepare.py --method char -t tokensfile.txt
  python3 ../prepare.py --method char -t input.txt --reuse_chars
  popd > /dev/null
}

for method in "${TRANSFORMS[@]}"; do
  build_variant "${method}"
done

python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets \
    databricks-dolly-15k \
    databricks-dolly-15k/mc_cvp \
    databricks-dolly-15k/mc_in_word_position \
    databricks-dolly-15k/mc_part_of_speech \
    databricks-dolly-15k/mc_since_newline \
    databricks-dolly-15k/mc_newlines_mod \
  --max_iters 2000 \
  --dropout 0.2 \
  --top_k 1 \
  --sample_each_eval \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --out_dir ./out_dolly_multicontext \
  --compile
