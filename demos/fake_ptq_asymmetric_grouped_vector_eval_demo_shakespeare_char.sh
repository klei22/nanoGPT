#!/bin/bash
# demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh
#
# Compares fake PTQ validation loss on shakespeare_char between:
#   1) Original full-vector PTQ (single quantization range per vector)
#   2) Grouped-vector asymmetric PTQ (independent range/zero-point per group)
#
# Default grouped setup: n_embd=300, 10 groups, 30 values per group.

set -euo pipefail

DATASET="shakespeare_char"
OUT_DIR="out_fake_ptq_${DATASET}_grouped_asym"
BASELINE_SWEEP_ROOT="${OUT_DIR}_baseline_vector_sweep"
GROUPED_SWEEP_ROOT="${OUT_DIR}_grouped_vector_asym_sweep"
EVAL_ROOT="${OUT_DIR}_evals"
SUMMARY_ROOT="${OUT_DIR}_quantization_summaries"

EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256
N_LAYER=4
N_HEAD=4
N_EMBD=300
MAX_ITERS=5000

VECTOR_GROUP_COUNT=10
VECTOR_GROUP_SIZE=0

BIT_START=8
BIT_STOP=3
BIT_STEP=-1

usage() {
  cat <<'USAGE'
Usage: demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh [options]

Options:
  --bit-start N           Starting bit-width (default: 8)
  --bit-stop N            Final bit-width (default: 3)
  --bit-step N            Sweep step (default: -1)
  --vector-group-count N  Number of groups per vector (default: 10)
  --vector-group-size N   Group size per vector (mutually exclusive with count)
  --help                  Show this help message and exit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bit-start)
      BIT_START="$2"
      shift 2
      ;;
    --bit-stop)
      BIT_STOP="$2"
      shift 2
      ;;
    --bit-step)
      BIT_STEP="$2"
      shift 2
      ;;
    --vector-group-count)
      VECTOR_GROUP_COUNT="$2"
      shift 2
      ;;
    --vector-group-size)
      VECTOR_GROUP_SIZE="$2"
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

if ! mapfile -t BITS < <(seq "$BIT_START" "$BIT_STEP" "$BIT_STOP"); then
  echo "Failed to generate bit-width sweep with start=$BIT_START, step=$BIT_STEP, stop=$BIT_STOP" >&2
  exit 1
fi

if [ "${#BITS[@]}" -eq 0 ]; then
  echo "Bit-width sweep is empty; adjust --bit-start/--bit-stop/--bit-step" >&2
  exit 1
fi

if [ "$VECTOR_GROUP_COUNT" -gt 0 ] && [ "$VECTOR_GROUP_SIZE" -gt 0 ]; then
  echo "--vector-group-count and --vector-group-size are mutually exclusive" >&2
  exit 1
fi

echo "Sweeping bits: ${BITS[*]}"
echo "Grouped-vector setup: n_embd=$N_EMBD, group_count=$VECTOR_GROUP_COUNT, group_size=$VECTOR_GROUP_SIZE"

mkdir -p "data/${DATASET}" "$BASELINE_SWEEP_ROOT" "$GROUPED_SWEEP_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT"

echo "=== Step 1: Prepare shakespeare_char dataset ==="
pushd "data/${DATASET}" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

echo "=== Step 2: Train reference model (if needed) ==="
if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset "$DATASET" \
    --out_dir "$OUT_DIR" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --n_embd "$N_EMBD" \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_iters "$MAX_ITERS" \
    --eval_interval 1000 \
    --eval_iters "$EVAL_ITERS" \
    --learning_rate 1e-3 \
    --compile
else
  echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate fp32 baseline ==="
BASELINE_EVAL_DIR="${EVAL_ROOT}/fp32"
mkdir -p "$BASELINE_EVAL_DIR"
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --eval_only \
  --eval_dataset "$DATASET" \
  --eval_iters "$EVAL_ITERS" \
  --eval_output_dir "$BASELINE_EVAL_DIR"

step=4
for bit in "${BITS[@]}"; do
  BASELINE_OUT="${BASELINE_SWEEP_ROOT}/${bit}bit"
  GROUPED_OUT="${GROUPED_SWEEP_ROOT}/${bit}bit"
  mkdir -p "$BASELINE_OUT" "$GROUPED_OUT"

  echo "=== Step ${step}: Quantize baseline full-vector PTQ (${bit}-bit, symmetric) ==="
  if [ ! -f "$BASELINE_OUT/ckpt.pt" ]; then
    python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
      --out_dir "$BASELINE_OUT" \
      --num_bits "$bit" \
      --granularity vector \
      --quantization symmetric
  else
    echo "Found existing baseline checkpoint at $BASELINE_OUT/ckpt.pt; skipping quantization."
  fi
  step=$((step + 1))

  echo "=== Step ${step}: Quantize grouped asymmetric vector PTQ (${bit}-bit) ==="
  if [ ! -f "$GROUPED_OUT/ckpt.pt" ]; then
    quant_args=(
      --out_dir "$GROUPED_OUT"
      --num_bits "$bit"
      --granularity vector
      --quantization asymmetric
    )
    if [ "$VECTOR_GROUP_COUNT" -gt 0 ]; then
      quant_args+=(--vector-group-count "$VECTOR_GROUP_COUNT")
    fi
    if [ "$VECTOR_GROUP_SIZE" -gt 0 ]; then
      quant_args+=(--vector-group-size "$VECTOR_GROUP_SIZE")
    fi

    python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" "${quant_args[@]}"
  else
    echo "Found existing grouped checkpoint at $GROUPED_OUT/ckpt.pt; skipping quantization."
  fi
  step=$((step + 1))

  BASELINE_EVAL_BIT_DIR="${EVAL_ROOT}/baseline_vector/${bit}bit"
  GROUPED_EVAL_BIT_DIR="${EVAL_ROOT}/grouped_asymmetric_vector/${bit}bit"
  mkdir -p "$BASELINE_EVAL_BIT_DIR" "$GROUPED_EVAL_BIT_DIR"

  echo "=== Step ${step}: Evaluate baseline full-vector PTQ (${bit}-bit) ==="
  python3 sample.py \
    --out_dir "$BASELINE_OUT" \
    --eval_only \
    --eval_dataset "$DATASET" \
    --eval_iters "$EVAL_ITERS" \
    --eval_output_dir "$BASELINE_EVAL_BIT_DIR"
  step=$((step + 1))

  echo "=== Step ${step}: Evaluate grouped asymmetric vector PTQ (${bit}-bit) ==="
  python3 sample.py \
    --out_dir "$GROUPED_OUT" \
    --eval_only \
    --eval_dataset "$DATASET" \
    --eval_iters "$EVAL_ITERS" \
    --eval_output_dir "$GROUPED_EVAL_BIT_DIR"
  step=$((step + 1))
done

python3 - "$EVAL_ROOT" "$SUMMARY_ROOT" "$VECTOR_GROUP_COUNT" "$VECTOR_GROUP_SIZE" --bits "${BITS[@]}" <<'PY'
import argparse
import csv
import json
import os
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("eval_root")
parser.add_argument("summary_root")
parser.add_argument("vector_group_count", type=int)
parser.add_argument("vector_group_size", type=int)
parser.add_argument("--bits", nargs="+", type=int, required=True)
args = parser.parse_args()

eval_root = os.path.abspath(args.eval_root)
summary_root = os.path.abspath(args.summary_root)
os.makedirs(summary_root, exist_ok=True)

def read_val_loss(path: str) -> float:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    val = data.get("val")
    if val is None:
        raise SystemExit(f"Missing 'val' in {path}")
    return float(val)

fp32_path = os.path.join(eval_root, "fp32", "eval_loss.txt")
if not os.path.exists(fp32_path):
    raise SystemExit(f"Missing baseline fp32 eval file: {fp32_path}")

fp32_val = read_val_loss(fp32_path)
rows = []
for bit in args.bits:
    base_eval = os.path.join(eval_root, "baseline_vector", f"{bit}bit", "eval_loss.txt")
    grouped_eval = os.path.join(eval_root, "grouped_asymmetric_vector", f"{bit}bit", "eval_loss.txt")
    if not os.path.exists(base_eval):
        raise SystemExit(f"Missing baseline eval for {bit}-bit: {base_eval}")
    if not os.path.exists(grouped_eval):
        raise SystemExit(f"Missing grouped eval for {bit}-bit: {grouped_eval}")

    base_val = read_val_loss(base_eval)
    grouped_val = read_val_loss(grouped_eval)
    rows.append({
        "bit": bit,
        "fp32_val_loss": fp32_val,
        "baseline_vector_val_loss": base_val,
        "grouped_asymmetric_vector_val_loss": grouped_val,
        "baseline_delta_vs_fp32": base_val - fp32_val,
        "grouped_delta_vs_fp32": grouped_val - fp32_val,
        "grouped_minus_baseline": grouped_val - base_val,
    })

csv_path = os.path.join(summary_root, "grouped_asymmetric_vector_eval_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

best_base = min(rows, key=lambda r: r["baseline_vector_val_loss"])
best_grouped = min(rows, key=lambda r: r["grouped_asymmetric_vector_val_loss"])
mean_gap = statistics.mean(r["grouped_minus_baseline"] for r in rows)

report_path = os.path.join(summary_root, "grouped_asymmetric_vector_eval_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("Grouped asymmetric vector PTQ vs baseline full-vector PTQ\n")
    fh.write(f"Bits swept: {[r['bit'] for r in rows]}\n")
    fh.write(f"FP32 validation loss: {fp32_val:.6f}\n")
    fh.write(
        f"Best baseline full-vector PTQ: {best_base['bit']}-bit @ {best_base['baseline_vector_val_loss']:.6f}\n"
    )
    fh.write(
        "Best grouped asymmetric vector PTQ: "
        f"{best_grouped['bit']}-bit @ {best_grouped['grouped_asymmetric_vector_val_loss']:.6f}\n"
    )
    fh.write(f"Mean(grouped - baseline) across sweep: {mean_gap:+.6f}\n")
    fh.write(
        "Grouped configuration: "
        f"vector_group_count={args.vector_group_count}, vector_group_size={args.vector_group_size}\n"
    )

print(f"Wrote CSV summary to {csv_path}")
print(f"Wrote text report to {report_path}")
PY

echo "=== Done ==="
echo "Summary CSV: ${SUMMARY_ROOT}/grouped_asymmetric_vector_eval_summary.csv"
echo "Summary report: ${SUMMARY_ROOT}/grouped_asymmetric_vector_eval_report.txt"
