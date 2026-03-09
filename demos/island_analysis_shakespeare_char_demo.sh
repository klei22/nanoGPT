#!/bin/bash
# Run island analysis on a Shakespeare-char checkpoint and generate summary + plotly dashboard.

set -euo pipefail

SKIP_TRAINING="${1:-yes}"
OUT_DIR="out_shakespeare_island_demo"
ANALYSIS_DIR="${OUT_DIR}/island_analysis"

bash data/shakespeare_char/get_dataset.sh

if [[ "${SKIP_TRAINING}" == "no" ]]; then
  rm -rf "${OUT_DIR}"
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${OUT_DIR}" \
    --max_iters 600 \
    --eval_interval 100 \
    --log_interval 10 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256
fi

if [[ ! -f "${OUT_DIR}/ckpt.pt" ]]; then
  echo "Missing checkpoint: ${OUT_DIR}/ckpt.pt" >&2
  echo "Run with: bash demos/island_analysis_shakespeare_char_demo.sh no" >&2
  exit 1
fi

python3 analysis/checkpoint_analysis/analyze_dot_islands_ckpt.py \
  "${OUT_DIR}" \
  --pattern "transformer\\.(wte|h\\.[0-9]+\\.(attn|mlp))" \
  --metric cosine \
  --thresholds 0.2,0.3,0.4,0.5 \
  --min_island_size 4 \
  --top_providers 8 \
  --out_dir "${ANALYSIS_DIR}"

echo "Done. See:"
echo "  - ${ANALYSIS_DIR}/islands_detailed.json"
echo "  - ${ANALYSIS_DIR}/islands_summary.csv"
echo "  - ${ANALYSIS_DIR}/islands_dashboard.html"
