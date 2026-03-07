#!/usr/bin/env bash
set -euo pipefail

# Demo: train two shakespeare_char models (n_embd 384 vs 320), then compare
# immediate next-token histograms by sweeping every possible input token.

OUT_BASE="${OUT_BASE:-out/immediate_hist_kl_demo}"
OUT_A="${OUT_BASE}/model_a_embd384"
OUT_B="${OUT_BASE}/model_b_embd320"
COMPARE_DIR="${OUT_BASE}/comparison"

MAX_ITERS="${MAX_ITERS:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

COMMON_ARGS=(
  --dataset shakespeare_char
  --max_iters "${MAX_ITERS}"
  --eval_interval "${EVAL_INTERVAL}"
  --block_size "${BLOCK_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --n_layer 6
  --n_head 3
  --n_qk_head_dim 100
  --n_v_head_dim 100
  --use_concat_heads
  --attention_variant infinite
  --n_kv_group 1
  --use_qk_norm
  --use_qk_norm_scale
  --use_pre_ln
  --use_peri_ln
  --no-use_post_ln
  --use_rotary_embeddings
  --no-use_abs_pos_embeddings
  --activation_variant squared_relu
  --softmax_variant_attn softmax
  --norm_variant_wte rmsnorm
  --compile
  --device "${DEVICE}"
)

echo "[1/3] Training model A (n_embd=384)"
python3 train.py \
  --out_dir "${OUT_A}" \
  --n_embd 384 \
  "${COMMON_ARGS[@]}"

echo "[2/3] Training model B (n_embd=320)"
python3 train.py \
  --out_dir "${OUT_B}" \
  --n_embd 320 \
  "${COMMON_ARGS[@]}"

echo "[3/3] Running first-association histogram + KL analysis"
python3 analysis/compare_first_association.py \
  --model_a_out_dir "${OUT_A}" \
  --model_b_out_dir "${OUT_B}" \
  --input_tokens_yaml all \
  --output_dir "${COMPARE_DIR}" \
  --top_k 20 \
  --batch_size 256 \
  --device "${DEVICE}"


echo "[4/4] Building interactive Plotly top-k comparison page"
python3 analysis/plot_first_association_pairs.py \
  --model_a_probs_yaml "${COMPARE_DIR}/model_a_probs.yaml" \
  --model_b_probs_yaml "${COMPARE_DIR}/model_b_probs.yaml" \
  --output_html "${COMPARE_DIR}/topk_next_token_pairs.html" \
  --output_json "${COMPARE_DIR}/topk_next_token_pairs.json" \
  --top_k 20

echo "Done. Artifacts are in: ${COMPARE_DIR}"
echo " - topk_logit_hist_side_by_side.png"
echo " - per_token_kl_barh.png"
echo " - per_token_kl.npy"
echo " - summary.json"
echo " - model_a_probs.yaml"
echo " - model_b_probs.yaml"
echo " - topk_next_token_pairs.html"
echo " - topk_next_token_pairs.json"
