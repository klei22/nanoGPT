#!/bin/bash
# demos/vector_lut_quant_demo.sh
#
# Evaluate the LUT-based vector quantization scheme across multiple hypersphere
# constructions and lookup-table sizes. The script trains a reference model on
# the minipile dataset (if needed), quantizes the checkpoint using
# `vector_lut_quantize_ckpt.py`, evaluates each quantized checkpoint, and plots
# validation loss versus LUT size for every construction.

set -euo pipefail

EVAL_DATASET_DIR="data/minipile"
OUT_DIR="out_vector_lut_minipile"
SWEEP_ROOT="${OUT_DIR}_lut_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256
VECTOR_DIM=384
LUT_METHODS=("grid_kronecker" "grid_halton" "lattice_rseq" "lattice_halton" "gaussian_baseline")
LUT_SIZES=(10000 100000 1000000)
SEED=42
GAUSSIAN_STD=0.02

mkdir -p "$EVAL_DATASET_DIR" "$SWEEP_ROOT"

echo "=== Step 1: Prepare the minipile dataset ==="
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

echo "=== Step 2: Train a reference model on minipile (if needed) ==="
if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset minipile \
    --out_dir "$OUT_DIR" \
    --n_layer 6 \
    --n_head 6 \
    --n_embd "$VECTOR_DIM" \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --use_qk_norm \
    --use_qk_norm_scale \
    --use_peri_ln \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_iters 10000 \
    --eval_interval 10000 \
    --eval_iters "$EVAL_ITERS" \
    --eta_variant iteration \
    --compile
else
  echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate the baseline (fp32) checkpoint ==="
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --eval_only \
  --eval_dataset minipile

step=4

for method in "${LUT_METHODS[@]}"; do
  if [ "$method" = "gaussian_baseline" ]; then
    echo "--- Highlighting gaussian baseline (mean=0.0, std=${GAUSSIAN_STD}) ---"
  else
    echo "--- Method: $method ---"
  fi
  for size in "${LUT_SIZES[@]}"; do
    QUANT_OUT_DIR="${SWEEP_ROOT}/${method}/${size}"
    mkdir -p "$QUANT_OUT_DIR"

    echo "=== Step ${step}: Quantize using ${method} with ${size} vectors ==="
    if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
      python3 quantizations/vector_lut_quantize_ckpt.py "$OUT_DIR" \
        --out_dir "$QUANT_OUT_DIR" \
        --method "$method" \
        --lut-size "$size" \
        --vector-dim "$VECTOR_DIM" \
        --chunk-size 65536 \
        --seed "$SEED" \
        --gaussian-std "$GAUSSIAN_STD"
    else
      echo "Found existing LUT-quantized checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
    fi

    step=$((step + 1))

    echo "=== Step ${step}: Evaluate ${method} (${size} vectors) ==="
    python3 sample.py \
      --out_dir "$QUANT_OUT_DIR" \
      --eval_only \
      --eval_dataset minipile

    step=$((step + 1))
  done
done

export LUT_METHODS_STR="${LUT_METHODS[*]}"
export LUT_SIZES_STR="${LUT_SIZES[*]}"
export GAUSSIAN_STD="$GAUSSIAN_STD"

python3 - "$OUT_DIR" "$SWEEP_ROOT" <<'PY'
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    plt = None
    print("matplotlib is not installed; skipping plot generation.")

if len(sys.argv) != 3:
    raise SystemExit("Expected OUT_DIR and SWEEP_ROOT arguments")

out_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])
methods = os.environ.get("LUT_METHODS_STR", "").split()
if not methods:
    raise SystemExit("No LUT methods provided via LUT_METHODS_STR")
try:
    sizes = [int(x) for x in os.environ.get("LUT_SIZES_STR", "").split()]
except ValueError as exc:  # pragma: no cover - defensive
    raise SystemExit(f"Failed to parse LUT sizes: {exc}")
if not sizes:
    raise SystemExit("No LUT sizes provided via LUT_SIZES_STR")

def load_eval(path: str) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing evaluation summary at {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if "val" not in data:
        raise KeyError(f"No 'val' key found in {path}")
    return float(data["val"])

results = []
fp32_loss = load_eval(os.path.join(out_dir, "eval_loss.txt"))
results.append({
    "method": "fp32",
    "lut_size": 0,
    "val_loss": fp32_loss,
})

for method in methods:
    for size in sizes:
        eval_path = os.path.join(sweep_root, method, str(size), "eval_loss.txt")
        val_loss = load_eval(eval_path)
        results.append({
            "method": method,
            "lut_size": size,
            "val_loss": val_loss,
        })

csv_path = os.path.join(sweep_root, "lut_quantization_eval.csv")
with open(csv_path, "w", encoding="utf-8") as csv_file:
    csv_file.write("method,lut_size,val_loss\n")
    for item in results:
        csv_file.write(f"{item['method']},{item['lut_size']},{item['val_loss']:.8f}\n")
print(f"Wrote summary CSV to {csv_path}")

if plt is not None:
    plt.figure(figsize=(9, 5))
    for method in methods:
        xs = []
        ys = []
        for size in sizes:
            xs.append(size)
            eval_path = os.path.join(sweep_root, method, str(size), "eval_loss.txt")
            ys.append(load_eval(eval_path))
        plt.plot(xs, ys, marker="o", label=method.replace("_", " "))
    plt.axhline(fp32_loss, linestyle="--", color="tab:orange", label="fp32 baseline")
    plt.xscale("log")
    plt.xlabel("LUT size (log scale)")
    plt.ylabel("Validation loss")
    plt.title("Vector LUT quantization: validation loss vs LUT size")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(sweep_root, "lut_quantization_eval.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Wrote plot to {plot_path}")

summary_path = os.path.join(sweep_root, "lut_quantization_eval.json")
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
print(f"Wrote JSON summary to {summary_path}")
PY

echo "LUT quantization sweep complete. Results available in $SWEEP_ROOT."
