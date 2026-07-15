#!/usr/bin/env bash
set -euo pipefail

# Experiment: hold infinite-attention Q/K width fixed at 250 and compare final
# token-embedding angle graphs while sweeping value-head width from 25 to 250.
#
# Override knobs, for example:
#   MAX_ITERS=2000 DEVICE=cuda V_DIMS="25 50 100 150 200 250" \
#     bash demos/infinite_value_dim_angle_graph_sweep.sh

OUT_BASE="${OUT_BASE:-out/infinite_vdim_angle_graph_sweep}"
DATASET="${DATASET:-shakespeare_char}"
META_PATH="${META_PATH:-data/shakespeare_char/meta.pkl}"
V_DIMS="${V_DIMS:-25 50 75 100 125 150 175 200 225 250}"
MAX_ITERS="${MAX_ITERS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_LAYER="${N_LAYER:-6}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-4}"
MAX_ANGLE_DEG="${MAX_ANGLE_DEG:-}"
COMPILE_FLAG="${COMPILE_FLAG:---compile}"

mkdir -p "${OUT_BASE}"

if [[ "${DATASET}" == "shakespeare_char" ]]; then
  bash data/shakespeare_char/get_dataset.sh
fi

COMMON_TRAIN_ARGS=(
  --dataset "${DATASET}"
  --max_iters "${MAX_ITERS}"
  --eval_interval "${EVAL_INTERVAL}"
  --eval_iters "${EVAL_ITERS}"
  --block_size "${BLOCK_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --n_layer "${N_LAYER}"
  --n_head 3
  --n_embd 250
  --attention_variant infinite
  --use_concat_heads
  --n_qk_head_dim 250
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
  --device "${DEVICE}"
)

if [[ -n "${COMPILE_FLAG}" ]]; then
  COMMON_TRAIN_ARGS+=("${COMPILE_FLAG}")
fi

INDEX="${OUT_BASE}/index.html"
cat > "${INDEX}" <<'HTML'
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Infinite attention value-dim angle graph sweep</title>
<style>body{font-family:system-ui,sans-serif;background:#020617;color:#e2e8f0;margin:2rem}a{color:#38bdf8}table{border-collapse:collapse}td,th{border:1px solid #334155;padding:.4rem .6rem}</style></head><body>
<h1>Infinite attention value-head dimension sweep</h1>
<p>Each row trains with <code>attention_variant=infinite</code>, <code>use_concat_heads</code>, <code>n_embd=250</code>, <code>n_qk_head_dim=250</code>, and <code>n_head=3</code>, then exports the final token-embedding angle graph to the fast canvas viewer.</p>
<table><thead><tr><th>n_v_head_dim</th><th>Viewer</th><th>Graph CSVs</th><th>Checkpoint</th></tr></thead><tbody>
HTML

for v_dim in ${V_DIMS}; do
  RUN_DIR="${OUT_BASE}/v${v_dim}"
  GRAPH_DIR="${RUN_DIR}/angle_graph"
  CKPT="${RUN_DIR}/ckpt.pt"
  VIEWER="${GRAPH_DIR}/viewer.html"

  echo "[train] n_v_head_dim=${v_dim} -> ${RUN_DIR}"
  python3 train.py \
    --out_dir "${RUN_DIR}" \
    --n_v_head_dim "${v_dim}" \
    "${COMMON_TRAIN_ARGS[@]}"

  if [[ ! -f "${CKPT}" ]]; then
    echo "Expected checkpoint not found: ${CKPT}" >&2
    exit 1
  fi

  echo "[graph] exporting token-embedding angle graph for n_v_head_dim=${v_dim}"
  graph_cmd=(
    python3 demos/ckpt_embedding_angle_graph.py
    --ckpt "${CKPT}"
    --meta "${META_PATH}"
    --output-dir "${GRAPH_DIR}"
    --top-k "${TOP_K}"
  )
  if [[ -n "${MAX_ANGLE_DEG}" ]]; then
    graph_cmd+=(--max-angle-deg "${MAX_ANGLE_DEG}")
  fi
  "${graph_cmd[@]}"

  echo "[viewer] precompiling fast HTML viewer for n_v_head_dim=${v_dim}"
  python3 demos/min_angle_graph_fast_viewer_demo.py \
    --adjacency-csv "${GRAPH_DIR}/adjacency.csv" \
    --token-list-csv "${GRAPH_DIR}/token_list.csv" \
    --dictionary-json "${GRAPH_DIR}/dictionary.json" \
    --output-html "${VIEWER}"

  cat >> "${INDEX}" <<HTML
<tr><td>${v_dim}</td><td><a href="v${v_dim}/angle_graph/viewer.html">open viewer</a></td><td><a href="v${v_dim}/angle_graph/adjacency.csv">adjacency</a> · <a href="v${v_dim}/angle_graph/token_list.csv">tokens</a></td><td><code>v${v_dim}/ckpt.pt</code></td></tr>
HTML
done

cat >> "${INDEX}" <<'HTML'
</tbody></table></body></html>
HTML

echo "Done. Open ${INDEX} to compare all final graphs."
