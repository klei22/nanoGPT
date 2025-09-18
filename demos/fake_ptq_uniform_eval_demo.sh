#!/bin/bash
# demos/fake_ptq_uniform_eval_demo.sh
#
# Runs the fake PTQ pipeline across a sweep of uniform bit-widths and records
# validation loss for each configuration. The script prepares the Shakespeare
# dataset, trains a compact reference model (if necessary), quantizes the
# checkpoint at bit-widths from 8 down to 1, evaluates each checkpoint with
# `train.py --eval_only`, and plots the resulting validation loss curve.

set -euo pipefail

EVAL_DATASET_DIR="data/shakespeare_char"
OUT_DIR="out_fake_ptq_shakespeare"
SWEEP_ROOT="${OUT_DIR}_uniform_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=128
BITS=(8 7 6 5 4 3 2 1)

mkdir -p "$EVAL_DATASET_DIR"

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "$SWEEP_ROOT"

echo "=== Step 2: Train a reference model on shakespeare_char (if needed) ==="
if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "$OUT_DIR" \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_iters 500 \
    --lr_decay_iters 500 \
    --eval_iters "$EVAL_ITERS" \
    --log_interval 10 \
    --always_save_checkpoint
else
  echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate the baseline (fp32) checkpoint ==="
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "$OUT_DIR" \
  --block_size "$BLOCK_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --init_from resume \
  --eval_only \
  --eval_iters "$EVAL_ITERS"

step=4
for bit in "${BITS[@]}"; do
  QUANT_OUT_DIR="${SWEEP_ROOT}/${bit}bit"
  mkdir -p "$QUANT_OUT_DIR"

  echo "=== Step ${step}: Quantize to ${bit}-bit weights ==="
  if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
    python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
      --out_dir "$QUANT_OUT_DIR" \
      --num_bits "$bit" \
      --quantization asymmetric
  else
    echo "Found existing ${bit}-bit checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
  fi

  step=$((step + 1))

  echo "=== Step ${step}: Evaluate the ${bit}-bit checkpoint ==="
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "$QUANT_OUT_DIR" \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --init_from resume \
    --eval_only \
    --eval_iters "$EVAL_ITERS"

  step=$((step + 1))
done

python3 - "$OUT_DIR" "$SWEEP_ROOT" <<'PY'
import json
import os
import sys

out_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])

entries = [("fp32", 32, os.path.join(out_dir, "eval_loss.txt"))]
for bit in range(8, 0, -1):
    entries.append((f"{bit}-bit", bit, os.path.join(sweep_root, f"{bit}bit", "eval_loss.txt")))

results = []
for label, bit, path in entries:
    if not os.path.exists(path):
        raise SystemExit(f"Missing evaluation summary at {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    val = data.get("val")
    if val is None:
        raise SystemExit(f"No 'val' key found in {path}")
    results.append((bit, float(val), label))

results.sort(key=lambda item: item[0], reverse=True)

csv_path = os.path.join(sweep_root, "uniform_quantization_eval.csv")
with open(csv_path, "w", encoding="utf-8") as csv_file:
    csv_file.write("bits,label,val_loss\n")
    for bit, loss, label in results:
        csv_file.write(f"{bit},{label},{loss:.8f}\n")
print(f"Wrote summary CSV to {csv_path}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping plot generation.")
else:
    bits = [item[0] for item in results]
    losses = [item[1] for item in results]
    labels = [item[2] for item in results]

    plt.figure(figsize=(8, 4.5))
    plt.plot(bits, losses, marker="o")
    plt.gca().invert_xaxis()
    plt.xticks(bits, labels)
    plt.xlabel("Uniform weight bit-width")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs quantization bit-width")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plot_path = os.path.join(sweep_root, "uniform_quantization_eval.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Wrote plot to {plot_path}")
PY

echo "Sweep complete. Evaluation summaries live in $SWEEP_ROOT."
