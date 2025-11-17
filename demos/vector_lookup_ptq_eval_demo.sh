#!/bin/bash
# demos/vector_lookup_ptq_eval_demo.sh
#
# Run the vector-lookup PTQ pipeline using several hyperspherical codebooks.
# For each construction (derived from the hypersphere grid and lattice
# analysis modules) sweep over three codebook sizes and evaluate the resulting
# validation loss on the Shakespeare character dataset.

set -euo pipefail

EVAL_DATASET_DIR="data/shakespeare_char"
OUT_DIR="out_fake_ptq_shakespeare"
SWEEP_ROOT="${OUT_DIR}_lookup_ptq_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256

DIM=384
SEED=42
CHUNK_SIZE=65536

METHODS=(
  grid_kronecker
  grid_halton
  grid_random
  lattice_rseq
  lattice_halton
  lattice_random
  gaussian_baseline
)

VECTOR_COUNTS=(10000 100000 1000000)

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
    --n_layer 6 \
    --n_head 6 \
    --n_embd "$DIM" \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_iters 750 \
    --eval_iters "$EVAL_ITERS" \
    --compile
else
  echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate the baseline (fp32) checkpoint ==="
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --eval_only \
  --eval_dataset shakespeare_char

step=4
for method in "${METHODS[@]}"; do
  for count in "${VECTOR_COUNTS[@]}"; do
    SUBDIR="${SWEEP_ROOT}/${method}/N${count}"
    mkdir -p "$SUBDIR"

    echo "=== Step ${step}: Quantize with method=${method}, vectors=${count} ==="
    if [ ! -f "$SUBDIR/ckpt.pt" ]; then
      python3 quantizations/ptq/vector_lookup_quantize_ckpt.py "$OUT_DIR" \
        --out_dir "$SUBDIR" \
        --method "$method" \
        --num_vectors "$count" \
        --dim "$DIM" \
        --seed "$SEED" \
        --chunk_size "$CHUNK_SIZE" \
        --save_codebook
    else
      echo "Found existing quantized checkpoint at $SUBDIR/ckpt.pt; skipping quantization."
    fi

    step=$((step + 1))

    echo "=== Step ${step}: Evaluate method=${method}, vectors=${count} ==="
    python3 sample.py \
      --out_dir "$SUBDIR" \
      --eval_only \
      --eval_dataset shakespeare_char

    step=$((step + 1))
  done
done

python3 - "$OUT_DIR" "$SWEEP_ROOT" <<'PY'
import json
import math
import os
import sys

out_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])

baseline_path = os.path.join(out_dir, "eval_loss.txt")
if not os.path.exists(baseline_path):
    raise SystemExit(f"Missing baseline evaluation at {baseline_path}")

with open(baseline_path, encoding="utf-8") as fh:
    baseline = float(json.load(fh)["val"])

records = []

for method in sorted(os.listdir(sweep_root)):
    method_dir = os.path.join(sweep_root, method)
    if not os.path.isdir(method_dir):
        continue
    for entry in sorted(os.listdir(method_dir)):
        if not entry.startswith("N"):
            continue
        try:
            num = int(entry[1:])
        except ValueError:
            continue
        eval_path = os.path.join(method_dir, entry, "eval_loss.txt")
        if not os.path.exists(eval_path):
            raise SystemExit(f"Missing evaluation summary at {eval_path}")
        with open(eval_path, encoding="utf-8") as fh:
            val = float(json.load(fh)["val"])
        records.append({
            "method": method,
            "num_vectors": num,
            "val_loss": val,
        })

records.sort(key=lambda r: (r["method"], r["num_vectors"]))

csv_path = os.path.join(sweep_root, "lookup_ptq_eval.csv")
with open(csv_path, "w", encoding="utf-8") as csv_file:
    csv_file.write("method,num_vectors,val_loss\n")
    for rec in records:
        csv_file.write(f"{rec['method']},{rec['num_vectors']},{rec['val_loss']:.8f}\n")
print(f"Wrote summary CSV to {csv_path}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping plot generation.")
    raise SystemExit(0)

plt.figure(figsize=(10, 6))
methods = sorted({rec["method"] for rec in records})
for method in methods:
    subset = [rec for rec in records if rec["method"] == method]
    subset.sort(key=lambda r: r["num_vectors"])
    xs = [rec["num_vectors"] for rec in subset]
    ys = [rec["val_loss"] for rec in subset]
    if not xs:
        continue
    plt.plot(xs, ys, marker="o", label=method)

plt.axhline(baseline, linestyle="--", color="black", label="fp32 baseline")
plt.xscale("log")
plt.xlabel("Codebook size (log scale)")
plt.ylabel("Validation loss")
plt.title("Vector lookup PTQ: validation loss vs codebook size")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plot_path = os.path.join(sweep_root, "lookup_ptq_eval.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
print(f"Wrote plot to {plot_path}")
PY

echo "=== Vector lookup PTQ evaluation complete ==="
