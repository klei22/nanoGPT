#!/usr/bin/env bash
# Optional two-stream route; does not change the legacy digits experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
read -r -a variants <<< "${EMBEDDING_MODES:-table_free table_sphere great_circle small_circle}"
read -r -a seeds <<< "${SEEDS:-0}"
args=(--variants "${variants[@]}" --seeds "${seeds[@]}"
  --steps "${MAX_ITERS:-2000}" --digits "${NUM_DIGITS:-8}"
  --digit-slots "${DIGIT_SLOTS:-10}" --letters "${NUM_LETTERS:-5}"
  --heads "${NUM_HEADS:-1}" --block-size "${BLOCK_SIZE:-16}"
  --batch-size "${BATCH_SIZE:-16}" --snapshot-every "${SAVE_INTERVAL:-1}"
  --learning-rate "${LEARNING_RATE:-0.003}" --weight-decay "${ADAM_WEIGHT_DECAY:-0.01}"
  --circle-offset "${CIRCLE_OFFSET:-0.5}" --device "${DEVICE:-cpu}"
  --embedding-init "${EMBEDDING_INIT:-random}" --pairing "${PAIRING:-aligned}"
  --output-dir "${DUAL_STREAM_DIR:-report/threejs/digits-3d/dual-stream}"
  --checkpoint-dir "${OUT_DIR:-out/dual_stream_clock}")
if [ -n "${WTE_FIXED_NORM_VALUE:-}" ]; then args+=(--radius "$WTE_FIXED_NORM_VALUE"); fi
python3 analysis/dual_stream_clock.py "${args[@]}" "$@"
cat <<'EOF'
Serve from the repository root:
  python3 -m http.server 8000
Open:
  http://localhost:8000/report/threejs/digits-3d/dual-stream.html
If DUAL_STREAM_DIR was customized, pass ?manifest=<path relative to dual-stream.html>.
EOF
