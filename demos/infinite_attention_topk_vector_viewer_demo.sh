#!/usr/bin/env bash
set -euo pipefail

# Demonstrates the dataset analyzer's saved top-k vector table and HTML viewer
# for checkpoints configured with variations/attention_variations.py's
# InfiniteHeadAttention (attention_variant=infinite). The prompt dataset contains
# one singular/plural English category pair per line.
#
# Usage:
#   OUT_DIR=out/infinite_attention_run DATASET=singular_plural_demo bash demos/infinite_attention_topk_vector_viewer_demo.sh

OUT_DIR=${OUT_DIR:-out/infinite_attention_run}
DATASET=${DATASET:-singular_plural_demo}
SPLIT=${SPLIT:-val}
DEVICE=${DEVICE:-cpu}
NUM_TOKENS=${NUM_TOKENS:-64}
TOPK=${TOPK:-5}

mkdir -p "data/${DATASET}"
cat > "data/${DATASET}/prompt.txt" <<'TXT'
cat cats
dog dogs
horse horses
apple apples
orange oranges
car cars
truck trucks
book books
TXT

cat <<'NOTE'
Created data/${DATASET}/prompt.txt. Tokenize it with this repository's normal
prepare flow for your tokenizer/checkpoint if train.bin/val.bin/meta.pkl do not
already exist. Then run the analyzer command below.
NOTE

python analyze_with_dataset.py \
  --out_dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --dtype float32 \
  --display topk \
  --topk "${TOPK}" \
  --num_tokens "${NUM_TOKENS}" \
  --window rolling \
  --activation_view rank_word \
  --components wte attn mlp resid \
  --save_topk_json "analysis_outputs/${DATASET}_topk_vectors.json" \
  --topk_viewer_file "${DATASET}_topk_vectors.html" \
  --save_attention_head_cproj \
  --saved_vector_components wte_raw wte_norm ln_f_input ln_f_output attn mlp attn_head_cproj

printf '\nOpen analysis_outputs/%s_topk_vectors.html in a browser.\n' "${DATASET}"
