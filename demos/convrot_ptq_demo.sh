#!/usr/bin/env bash
# Educational ConvRot fake-W4A4 demo for a small nanoGPT checkpoint.
set -euo pipefail

OUT_DIR="${OUT_DIR:-out_convrot_ptq_demo}"
GROUP_SIZE="${GROUP_SIZE:-16}"

if [[ ! -f data/shakespeare_char/train.bin ]]; then
  python3 data/shakespeare_char/prepare.py
fi

if [[ ! -f "$OUT_DIR/ckpt.pt" ]]; then
  python3 train.py \
    --dataset shakespeare_char --out_dir "$OUT_DIR" --device cpu \
    --dtype float32 --compile false --eval_interval 10 --eval_iters 5 \
    --log_interval 1 --block_size 64 --batch_size 8 --n_layer 2 \
    --n_head 2 --n_embd 64 --max_iters 10
fi

python3 quantizations/ptq/convrot_demo.py "$OUT_DIR/ckpt.pt" \
  --tensor 'attn.*weight' \
  --group-size "$GROUP_SIZE" \
  --output "$OUT_DIR/convrot_report.json"

echo "ConvRot report written to $OUT_DIR/convrot_report.json"
