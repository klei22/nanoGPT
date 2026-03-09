#!/bin/bash
# demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh
#
# Compares fake PTQ validation loss on shakespeare_char between:
#   1) Original full-vector PTQ (single quantization range per vector)
#   2) Grouped-vector asymmetric PTQ (independent range/zero-point per group)
#
# Defaults:
#   - n_embd=300
#   - bit sweep: int8 -> int3
#   - group-count sweep: 1 -> 10 (30 values/group at group_count=10)

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

BIT_START=8
BIT_STOP=3
BIT_STEP=-1

GROUP_COUNT_START=1
GROUP_COUNT_STOP=10
GROUP_COUNT_STEP=1

usage() {
  cat <<'USAGE'
Usage: demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh [options]

Options:
  --bit-start N            Starting bit-width (default: 8)
  --bit-stop N             Final bit-width (default: 3)
  --bit-step N             Sweep step (default: -1)
  --group-count-start N    Starting group-count (default: 1)
  --group-count-stop N     Final group-count (default: 10)
  --group-count-step N     Group-count sweep step (default: 1)
  --help                   Show this help message and exit
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
    --group-count-start)
      GROUP_COUNT_START="$2"
      shift 2
      ;;
    --group-count-stop)
      GROUP_COUNT_STOP="$2"
      shift 2
      ;;
    --group-count-step)
      GROUP_COUNT_STEP="$2"
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

if ! mapfile -t GROUP_COUNTS < <(seq "$GROUP_COUNT_START" "$GROUP_COUNT_STEP" "$GROUP_COUNT_STOP"); then
  echo "Failed to generate group-count sweep with start=$GROUP_COUNT_START, step=$GROUP_COUNT_STEP, stop=$GROUP_COUNT_STOP" >&2
  exit 1
fi

if [ "${#BITS[@]}" -eq 0 ]; then
  echo "Bit-width sweep is empty; adjust --bit-start/--bit-stop/--bit-step" >&2
  exit 1
fi

if [ "${#GROUP_COUNTS[@]}" -eq 0 ]; then
  echo "Group-count sweep is empty; adjust --group-count-start/--group-count-stop/--group-count-step" >&2
  exit 1
fi

for gc in "${GROUP_COUNTS[@]}"; do
  if [ "$gc" -le 0 ]; then
    echo "Invalid group count: $gc (must be > 0)" >&2
    exit 1
  fi
  if (( N_EMBD % gc != 0 )); then
    echo "Invalid group count: $gc does not divide n_embd=$N_EMBD" >&2
    exit 1
  fi
done

echo "Sweeping bits: ${BITS[*]}"
echo "Sweeping group-counts: ${GROUP_COUNTS[*]}"
echo "Model setup: n_embd=$N_EMBD"

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
  mkdir -p "$BASELINE_OUT"

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

  BASELINE_EVAL_BIT_DIR="${EVAL_ROOT}/baseline_vector/${bit}bit"
  mkdir -p "$BASELINE_EVAL_BIT_DIR"

  echo "=== Step ${step}: Evaluate baseline full-vector PTQ (${bit}-bit) ==="
  python3 sample.py \
    --out_dir "$BASELINE_OUT" \
    --eval_only \
    --eval_dataset "$DATASET" \
    --eval_iters "$EVAL_ITERS" \
    --eval_output_dir "$BASELINE_EVAL_BIT_DIR"
  step=$((step + 1))

  for group_count in "${GROUP_COUNTS[@]}"; do
    group_size=$((N_EMBD / group_count))
    GROUPED_OUT="${GROUPED_SWEEP_ROOT}/${bit}bit/groups_${group_count}"
    mkdir -p "$GROUPED_OUT"

    echo "=== Step ${step}: Quantize grouped asymmetric vector PTQ (${bit}-bit, groups=${group_count}, group_size=${group_size}) ==="
    if [ ! -f "$GROUPED_OUT/ckpt.pt" ]; then
      python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
        --out_dir "$GROUPED_OUT" \
        --num_bits "$bit" \
        --granularity vector \
        --quantization asymmetric \
        --vector-group-count "$group_count"
    else
      echo "Found existing grouped checkpoint at $GROUPED_OUT/ckpt.pt; skipping quantization."
    fi
    step=$((step + 1))

    GROUPED_EVAL_BIT_DIR="${EVAL_ROOT}/grouped_asymmetric_vector/${bit}bit/groups_${group_count}"
    mkdir -p "$GROUPED_EVAL_BIT_DIR"

    echo "=== Step ${step}: Evaluate grouped asymmetric vector PTQ (${bit}-bit, groups=${group_count}) ==="
    python3 sample.py \
      --out_dir "$GROUPED_OUT" \
      --eval_only \
      --eval_dataset "$DATASET" \
      --eval_iters "$EVAL_ITERS" \
      --eval_output_dir "$GROUPED_EVAL_BIT_DIR"
    step=$((step + 1))
  done
done

python3 - "$EVAL_ROOT" "$SUMMARY_ROOT" "$N_EMBD" --bits "${BITS[@]}" --group-counts "${GROUP_COUNTS[@]}" <<'PY'
import argparse
import csv
import json
import os
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("eval_root")
parser.add_argument("summary_root")
parser.add_argument("n_embd", type=int)
parser.add_argument("--bits", nargs="+", type=int, required=True)
parser.add_argument("--group-counts", nargs="+", type=int, required=True)
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
    if not os.path.exists(base_eval):
        raise SystemExit(f"Missing baseline eval for {bit}-bit: {base_eval}")

    base_val = read_val_loss(base_eval)
    for group_count in args.group_counts:
        grouped_eval = os.path.join(
            eval_root,
            "grouped_asymmetric_vector",
            f"{bit}bit",
            f"groups_{group_count}",
            "eval_loss.txt",
        )
        if not os.path.exists(grouped_eval):
            raise SystemExit(
                f"Missing grouped eval for {bit}-bit groups={group_count}: {grouped_eval}"
            )

        grouped_val = read_val_loss(grouped_eval)
        rows.append(
            {
                "bit": bit,
                "group_count": group_count,
                "group_size": args.n_embd // group_count,
                "fp32_val_loss": fp32_val,
                "baseline_vector_val_loss": base_val,
                "grouped_asymmetric_vector_val_loss": grouped_val,
                "baseline_delta_vs_fp32": base_val - fp32_val,
                "grouped_delta_vs_fp32": grouped_val - fp32_val,
                "grouped_minus_baseline": grouped_val - base_val,
            }
        )

if not rows:
    raise SystemExit("No rows collected for summary")

csv_path = os.path.join(summary_root, "grouped_asymmetric_vector_eval_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

agg_rows = []
for group_count in args.group_counts:
    subset = [r for r in rows if r["group_count"] == group_count]
    agg_rows.append(
        {
            "group_count": group_count,
            "group_size": args.n_embd // group_count,
            "best_grouped_val_loss": min(r["grouped_asymmetric_vector_val_loss"] for r in subset),
            "best_grouped_bit": min(subset, key=lambda r: r["grouped_asymmetric_vector_val_loss"])["bit"],
            "mean_grouped_minus_baseline": statistics.mean(r["grouped_minus_baseline"] for r in subset),
        }
    )

agg_csv_path = os.path.join(summary_root, "grouped_asymmetric_vector_eval_group_aggregate.csv")
with open(agg_csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(agg_rows[0].keys()))
    writer.writeheader()
    writer.writerows(agg_rows)

best_base_row = min(rows, key=lambda r: r["baseline_vector_val_loss"])
best_grouped_row = min(rows, key=lambda r: r["grouped_asymmetric_vector_val_loss"])

report_path = os.path.join(summary_root, "grouped_asymmetric_vector_eval_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("Grouped asymmetric vector PTQ vs baseline full-vector PTQ\n")
    fh.write(f"Bits swept: {args.bits}\n")
    fh.write(f"Group-counts swept: {args.group_counts}\n")
    fh.write(f"Embedding dimension: {args.n_embd}\n")
    fh.write(f"FP32 validation loss: {fp32_val:.6f}\n")
    fh.write(
        f"Best baseline full-vector PTQ: {best_base_row['bit']}-bit @ {best_base_row['baseline_vector_val_loss']:.6f}\n"
    )
    fh.write(
        "Best grouped asymmetric vector PTQ overall: "
        f"{best_grouped_row['bit']}-bit @ groups={best_grouped_row['group_count']} "
        f"(group_size={best_grouped_row['group_size']}) @ "
        f"{best_grouped_row['grouped_asymmetric_vector_val_loss']:.6f}\n"
    )

print(f"Wrote detailed CSV summary to {csv_path}")
print(f"Wrote group aggregate CSV to {agg_csv_path}")
print(f"Wrote text report to {report_path}")
PY

echo "=== Done ==="
echo "Summary CSV: ${SUMMARY_ROOT}/grouped_asymmetric_vector_eval_summary.csv"
echo "Aggregate CSV: ${SUMMARY_ROOT}/grouped_asymmetric_vector_eval_group_aggregate.csv"
echo "Summary report: ${SUMMARY_ROOT}/grouped_asymmetric_vector_eval_report.txt"
