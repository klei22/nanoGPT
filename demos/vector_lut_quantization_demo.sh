#!/bin/bash
# demos/vector_lut_quantization_demo.sh
#
# Demonstrate vector lookup table quantization using multiple hypersphere
# construction methods across several table sizes. The script prepares the
# Shakespeare character dataset, trains a reference model (if missing), applies
# vector-LUT quantization using generators inspired by the hypersphere grid and
# lattice utilities, evaluates the resulting checkpoints, and plots validation
# loss versus LUT size. A gaussian baseline (mean 0.0, stddev 0.02 prior to
# normalization) is included for comparison.

set -euo pipefail

EVAL_DATASET="shakespeare_char"
EVAL_DATASET_DIR="data/${EVAL_DATASET}"
BASE_OUT_DIR="out_vector_lut_${EVAL_DATASET}"
SWEEP_ROOT="${BASE_OUT_DIR}_vector_lut_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256
N_LAYER=6
N_HEAD=6
N_EMBD=384

LUT_METHODS=("kronecker" "halton" "rseq" "random_sphere" "gaussian_baseline")
LUT_SIZES=(10000 100000 1000000)

usage() {
  cat <<'USAGE'
Usage: demos/vector_lut_quantization_demo.sh [--methods m1,m2,...] [--sizes n1,n2,...]

  --methods  Comma separated list of LUT generators to sweep
             (default: kronecker,halton,rseq,random_sphere,gaussian_baseline)
  --sizes    Comma separated list of LUT sizes to evaluate (default: 10000,100000,1000000)
  --help     Show this help message and exit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods)
      IFS=',' read -r -a LUT_METHODS <<< "$2"
      shift 2
      ;;
    --sizes)
      IFS=',' read -r -a LUT_SIZES <<< "$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ "${#LUT_METHODS[@]}" -eq 0 ]; then
  echo "No LUT methods specified" >&2
  exit 1
fi
if [ "${#LUT_SIZES[@]}" -eq 0 ]; then
  echo "No LUT sizes specified" >&2
  exit 1
fi

mkdir -p "$EVAL_DATASET_DIR"

echo "=== Step 1: Prepare the ${EVAL_DATASET} dataset ==="
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "$SWEEP_ROOT"

echo "=== Step 2: Train or locate the reference model (n_embd=${N_EMBD}) ==="
if [ ! -f "$BASE_OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset "$EVAL_DATASET" \
    --out_dir "$BASE_OUT_DIR" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --n_embd "$N_EMBD" \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_iters 750 \
    --eval_iters "$EVAL_ITERS" \
    --compile
else
  echo "Found existing checkpoint at $BASE_OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate the baseline (fp32) checkpoint ==="
python3 sample.py \
  --out_dir "$BASE_OUT_DIR" \
  --eval_only \
  --eval_dataset "$EVAL_DATASET"

step=4
for method in "${LUT_METHODS[@]}"; do
  for size in "${LUT_SIZES[@]}"; do
    QUANT_OUT_DIR="${SWEEP_ROOT}/${method}/${size}"
    mkdir -p "$QUANT_OUT_DIR"

    echo "=== Step ${step}: Quantize using method='${method}' with ${size} vectors ==="
    if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
      python3 quantizations/vector_lut/vector_lut_quantize_ckpt.py "$BASE_OUT_DIR" \
        --out_dir "$QUANT_OUT_DIR" \
        --lut-method "$method" \
        --lut-size "$size" \
        --dim "$N_EMBD" \
        --seed 0 \
        --lut-out vector_lut.npy \
        --indices-out indices \
        --chunk-size 16384 \
        --row-chunk-size 512
    else
      echo "Found existing quantized checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
    fi

    step=$((step + 1))

    echo "=== Step ${step}: Evaluate method='${method}' (${size} vectors) ==="
    python3 sample.py \
      --out_dir "$QUANT_OUT_DIR" \
      --eval_only \
      --eval_dataset "$EVAL_DATASET"

    step=$((step + 1))
  done
done

python3 - "${BASE_OUT_DIR}" "${SWEEP_ROOT}" "${LUT_METHODS[@]}" -- "${LUT_SIZES[@]}" <<'PY'
import json
import math
import os
import sys

out_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])
args = sys.argv[3:]
if "--" not in args:
    raise SystemExit("Expected method/size separator '--'")
sep = args.index("--")
methods = args[:sep]
sizes = [int(x) for x in args[sep + 1 :]]

if not methods or not sizes:
    raise SystemExit("No methods or sizes provided to aggregation helper")

entries = []
with open(os.path.join(out_dir, "eval_loss.txt"), encoding="utf-8") as fh:
    baseline = json.load(fh)
entries.append({
    "method": "fp32",
    "lut_size": 0,
    "val_loss": float(baseline.get("val", float("nan"))),
    "mean_similarity": math.nan,
})

for method in methods:
    for size in sizes:
        quant_dir = os.path.join(sweep_root, method, str(size))
        eval_path = os.path.join(quant_dir, "eval_loss.txt")
        if not os.path.exists(eval_path):
            raise SystemExit(f"Missing evaluation summary: {eval_path}")
        with open(eval_path, encoding="utf-8") as fh:
            eval_data = json.load(fh)
        val_loss = float(eval_data.get("val", float("nan")))
        meta_path = os.path.join(quant_dir, "vector_lut_quantization.json")
        mean_similarity = math.nan
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            stats = meta.get("tensor_stats", {})
            sims = [float(info.get("mean_similarity", math.nan)) for info in stats.values()]
            sims = [s for s in sims if not math.isnan(s)]
            if sims:
                mean_similarity = sum(sims) / len(sims)
        entries.append({
            "method": method,
            "lut_size": size,
            "val_loss": val_loss,
            "mean_similarity": mean_similarity,
        })

entries.sort(key=lambda item: (item["method"], item["lut_size"]))

csv_path = os.path.join(sweep_root, "vector_lut_eval.csv")
with open(csv_path, "w", encoding="utf-8") as fh:
    fh.write("method,lut_size,val_loss,mean_similarity\n")
    for item in entries:
        fh.write(
            f"{item['method']},{item['lut_size']},{item['val_loss']:.8f},{item['mean_similarity'] if not math.isnan(item['mean_similarity']) else ''}\n"
        )
print(f"Wrote summary CSV to {csv_path}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping plot generation.")
    raise SystemExit(0)

plot_path = os.path.join(sweep_root, "vector_lut_eval.png")
plt.figure(figsize=(9, 5))

baseline_loss = next(item for item in entries if item["method"] == "fp32")
plt.axhline(
    baseline_loss["val_loss"],
    color="tab:gray",
    linestyle="--",
    linewidth=1.5,
    label="fp32 baseline",
)

palette = plt.get_cmap("tab10")
method_colors = {method: palette(idx % 10) for idx, method in enumerate(methods)}

for method in methods:
    xs = []
    ys = []
    for size in sizes:
        record = next((item for item in entries if item["method"] == method and item["lut_size"] == size), None)
        if record is None:
            continue
        xs.append(record["lut_size"])
        ys.append(record["val_loss"])
    if not xs:
        continue
    linestyle = "--" if method == "gaussian_baseline" else "-"
    plt.plot(
        xs,
        ys,
        marker="o",
        linestyle=linestyle,
        color=method_colors[method],
        label=method.replace("_", " ") + (" (gaussian)" if method == "gaussian_baseline" else ""),
    )

plt.xscale("log")
plt.xticks(sizes, [f"{size:,}" for size in sizes])
plt.xlabel("Lookup table size (log scale)")
plt.ylabel("Validation loss")
plt.title("Validation loss vs. vector lookup table size")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=200)
print(f"Saved evaluation plot to {plot_path}")
PY

