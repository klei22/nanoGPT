#!/bin/bash
# summary_token_opus100_demo.sh
# Demonstrates the summary token feature with the OPUS-100 translation dataset.
#
# Workflow:
#   1) Download and prepare an English-Spanish opus-100 translation pair dataset
#   2) Train a model with the summary token enabled (two-phase training)
#   3) Sample normally (baseline)
#   4) Sample using the summary token to compress the source text into a
#      summary vector, then generate the translation from that vector
#
# The summary token acts as a learned "bottleneck" that soaks up the meaning
# of the source text and can then be used to predict the target translation.

set -euo pipefail

FROM_LANG="en"
TO_LANG="es"
DATA_DIR="data/opus-100"
DATASET_NAME="opus-100"
OUT_DIR="out/summary_token_opus100_demo"
BLOCK_SIZE=256
N_PREFILL=128
MAX_ITERS=5000
EVAL_INTERVAL=500

echo "============================================================"
echo " Summary Token + OPUS-100 Translation Demo"
echo "============================================================"

# ── Step 1: Download and prepare the dataset ─────────────────────
echo ""
echo "=== Step 1: Prepare the opus-100 ${FROM_LANG}-${TO_LANG} dataset ==="
pushd "${DATA_DIR}" > /dev/null

if [ ! -f "input.txt" ]; then
  echo "Downloading opus-100 ${FROM_LANG}-${TO_LANG} data..."
  python3 get_dataset.py -f "${FROM_LANG}" -t "${TO_LANG}" -o input.txt
else
  echo "Found existing input.txt, skipping download."
fi

if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  echo "Tokenizing dataset..."
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "Found existing tokenized dataset artifacts."
fi

popd > /dev/null

mkdir -p "${OUT_DIR}"

# ── Step 2: Train with summary token ────────────────────────────
echo ""
echo "=== Step 2: Train with summary token enabled ==="
echo "  block_size=${BLOCK_SIZE}, n_prefill=${N_PREFILL}"
echo "  The model learns both next-token prediction AND summary-token compression."

python3 train.py \
  --dataset "${DATASET_NAME}" \
  --out_dir "${OUT_DIR}" \
  --block_size "${BLOCK_SIZE}" \
  --batch_size 16 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --max_iters "${MAX_ITERS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --eval_iters 100 \
  --learning_rate 6e-4 \
  --weight_decay 0.1 \
  --dropout 0.1 \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --use_summary_token \
  --summary_token_n_prefill "${N_PREFILL}" \
  --summary_token_loss_weight 1.0 \
  --max_sample_tokens 100 \
  --sample_start_tokens "The fire department has its own investigative unit." \
  --compile

echo ""
echo "Training complete. Checkpoint saved to ${OUT_DIR}/ckpt.pt"

# ── Step 3: Sample without summary token (baseline) ─────────────
echo ""
echo "=== Step 3: Baseline sampling (no summary token) ==="
python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --start "The fire department has its own investigative unit." \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 200 \
  --num_samples 2 \
  --device cuda

# ── Step 4: Sample WITH summary token ───────────────────────────
echo ""
echo "=== Step 4: Summary token sampling ==="
echo "  The start text is compressed into a summary vector, then generation"
echo "  proceeds from that single vector (translation from compressed context)."
python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --start "The fire department has its own investigative unit." \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 200 \
  --num_samples 2 \
  --device cuda \
  --use_summary_token

echo ""
echo "============================================================"
echo " Demo complete!"
echo " Compare baseline vs summary-token outputs above."
echo " The summary token learns to compress the source sentence"
echo " and use it as context for generating the continuation."
echo "============================================================"
