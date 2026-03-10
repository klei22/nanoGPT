#!/usr/bin/env bash
set -euo pipefail

# Demo: train three shakespeare_char models (n_embd 256/384/512), then compare
# first-association next-token behavior over all start tokens.

OUT_BASE="${OUT_BASE:-out/immediate_hist_kl_demo}"
OUT_256="${OUT_BASE}/model_embd256"
OUT_384="${OUT_BASE}/model_embd384"
OUT_512="${OUT_BASE}/model_embd512"
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

echo "[1/6] Training model embd256"
python3 train.py \
  --out_dir "${OUT_256}" \
  --n_embd 256 \
  "${COMMON_ARGS[@]}"

echo "[2/6] Training model embd384"
python3 train.py \
  --out_dir "${OUT_384}" \
  --n_embd 384 \
  "${COMMON_ARGS[@]}"

echo "[3/6] Training model embd512"
python3 train.py \
  --out_dir "${OUT_512}" \
  --n_embd 512 \
  "${COMMON_ARGS[@]}"

echo "[4/6] Running first-association histogram + KL analysis"
python3 analysis/compare_first_association.py \
  --model_out_dir "${OUT_256}" "${OUT_384}" "${OUT_512}" \
  --model_label embd256 embd384 embd512 \
  --reference_label embd384 \
  --input_tokens_yaml all \
  --output_dir "${COMPARE_DIR}" \
  --top_k 20 \
  --batch_size 256 \
  --device "${DEVICE}"

echo "[5/6] Building interactive Plotly top-k comparison page"
python3 analysis/plot_first_association_pairs.py \
  --probs_yaml "${COMPARE_DIR}/probs_embd256.yaml" "${COMPARE_DIR}/probs_embd384.yaml" "${COMPARE_DIR}/probs_embd512.yaml" \
  --label embd256 embd384 embd512 \
  --output_html "${COMPARE_DIR}/topk_next_token_pairs.html" \
  --output_json "${COMPARE_DIR}/topk_next_token_pairs.json" \
  --top_k 20


echo "[6/6] Building static PyVis network graph"
python3 analysis/first_association_pyvis_graph.py \
  --probs_yaml "${COMPARE_DIR}/probs_embd384.yaml" \
  --output_html "${COMPARE_DIR}/first_association_graph_embd384.html" \
  --backend auto \
  --top_k 12 \
  --edge_percentile_keep 85 \
  --max_edges 4000 \
  --min_strength 0.0 \
  --initial_start_tokens none \
  --node_selector_scope start

echo "Done. Artifacts are in: ${COMPARE_DIR}"
echo " - topk_logit_hist_by_model.png"
echo " - per_token_kl_barh.png"
echo " - per_token_kl_embd384_vs_embd256.npy"
echo " - per_token_kl_embd384_vs_embd512.npy"
echo " - summary.json"
echo " - probs_manifest.json"
echo " - probs_embd256.yaml"
echo " - probs_embd384.yaml"
echo " - probs_embd512.yaml"
echo " - topk_next_token_pairs.html"
echo " - topk_next_token_pairs.json"
echo " - first_association_graph_embd384.html"
