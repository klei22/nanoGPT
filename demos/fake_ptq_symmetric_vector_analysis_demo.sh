#!/bin/bash
# demos/fake_ptq_symmetric_vector_analysis_demo.sh
#
# Performs symmetrical per-vector post-training quantization analysis on a
# provided checkpoint directory.  Sweeps bit-widths int8 through int3,
# evaluates validation loss for each, and computes angular distortion between
# the original and quantized weight vectors.  Produces two plots:
#   1. Validation loss vs. bit-width
#   2. Mean angular distortion (degrees) vs. bit-width
# Results are written as CSV and PNG to a summary directory.
#
# Parallelization strategy (3 phases):
#   Phase 1: Baseline fp32 eval (GPU) runs concurrently with all 6 quantizations (CPU)
#   Phase 2: GPU evals run sequentially; CPU angle analyses run in background
#   Phase 3: Aggregate CSV + plot generation

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
CKPT_DIR=""
EVAL_DATASET=""
EVAL_ITERS=200
BIT_WIDTHS=(8 7 6 5 4 3)
SWEEP_ROOT=""
SUMMARY_ROOT=""

usage() {
  cat <<'EOF'
Usage: demos/fake_ptq_symmetric_vector_analysis_demo.sh --ckpt-dir <path> --eval-dataset <name> [OPTIONS]

Required:
  --ckpt-dir       Path to the directory containing the source ckpt.pt
  --eval-dataset   Name of the evaluation dataset (e.g. shakespeare_char)

Options:
  --eval-iters     Number of evaluation iterations (default: 200)
  --sweep-root     Directory for quantized checkpoints
                   (default: <ckpt-dir>_sym_vector_sweep)
  --summary-root   Directory for output CSV and plots
                   (default: <ckpt-dir>_sym_vector_summary)
  --help           Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --eval-dataset)
      EVAL_DATASET="$2"
      shift 2
      ;;
    --eval-iters)
      EVAL_ITERS="$2"
      shift 2
      ;;
    --sweep-root)
      SWEEP_ROOT="$2"
      shift 2
      ;;
    --summary-root)
      SUMMARY_ROOT="$2"
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

if [ -z "$CKPT_DIR" ]; then
  echo "Error: --ckpt-dir is required." >&2
  usage >&2
  exit 1
fi

if [ -z "$EVAL_DATASET" ]; then
  echo "Error: --eval-dataset is required." >&2
  usage >&2
  exit 1
fi

if [ ! -f "$CKPT_DIR/ckpt.pt" ]; then
  echo "Error: No ckpt.pt found in $CKPT_DIR" >&2
  exit 1
fi

# Derive output directories from checkpoint dir if not explicitly set
if [ -z "$SWEEP_ROOT" ]; then
  SWEEP_ROOT="${CKPT_DIR}_sym_vector_sweep"
fi
if [ -z "$SUMMARY_ROOT" ]; then
  SUMMARY_ROOT="${CKPT_DIR}_sym_vector_summary"
fi

EVAL_DATASET_DIR="data/${EVAL_DATASET}"
EVAL_ROOT="${SWEEP_ROOT}_evals"

echo "============================================================"
echo "Symmetric per-vector PTQ analysis"
echo "  Checkpoint:     $CKPT_DIR"
echo "  Eval dataset:   $EVAL_DATASET"
echo "  Eval iters:     $EVAL_ITERS"
echo "  Bit-widths:     ${BIT_WIDTHS[*]}"
echo "  Sweep root:     $SWEEP_ROOT"
echo "  Summary root:   $SUMMARY_ROOT"
echo "============================================================"

# ── Step 1: Verify dataset ──────────────────────────────────────────────────
echo ""
echo "=== Step 1: Verify evaluation dataset ==="
if [ ! -d "$EVAL_DATASET_DIR" ]; then
  echo "Error: Dataset directory $EVAL_DATASET_DIR does not exist." >&2
  echo "Please prepare the dataset before running this demo." >&2
  exit 1
fi

if [ ! -f "$EVAL_DATASET_DIR/val.bin" ]; then
  echo "Error: val.bin not found in $EVAL_DATASET_DIR" >&2
  echo "Please prepare the dataset before running this demo." >&2
  exit 1
fi
echo "Found evaluation dataset at $EVAL_DATASET_DIR"

mkdir -p "$SWEEP_ROOT" "$SUMMARY_ROOT" "$EVAL_ROOT"

# Regex for weight tensors to compare angles (attention + MLP weights)
PATTERN='transformer\.h\.[0-9]+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight'

# ── Phase 1: Baseline eval (GPU) + all quantizations (CPU) in parallel ──────
echo ""
echo "=== Phase 1: Baseline eval + all quantizations in parallel ==="

# Start baseline eval in background (uses GPU)
BASELINE_EVAL_DIR="${EVAL_ROOT}/fp32"
mkdir -p "$BASELINE_EVAL_DIR"
echo "  Starting baseline (fp32) evaluation..."
python3 sample.py \
  --out_dir "$CKPT_DIR" \
  --eval_only \
  --eval_dataset "$EVAL_DATASET" \
  --eval_iters "$EVAL_ITERS" \
  --eval_output_dir "$BASELINE_EVAL_DIR" &
BASELINE_PID=$!

# Launch all quantizations in parallel (CPU-only, no GPU contention)
QUANT_PIDS=()
for bit in "${BIT_WIDTHS[@]}"; do
  QUANT_OUT_DIR="${SWEEP_ROOT}/${bit}bit"
  mkdir -p "$QUANT_OUT_DIR"
  if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
    echo "  Starting int${bit} quantization..."
    python3 quantizations/ptq/fake_quantize_ckpt.py "$CKPT_DIR" \
      --out_dir "$QUANT_OUT_DIR" \
      --num_bits "$bit" \
      --quantization symmetric \
      --granularity vector &
    QUANT_PIDS+=($!)
  else
    echo "  Found existing int${bit} checkpoint; skipping quantization."
  fi
done

# Wait for baseline eval first (frees GPU for phase 2)
echo "  Waiting for baseline evaluation..."
wait "$BASELINE_PID" || { echo "Baseline evaluation failed" >&2; exit 1; }
echo "  Baseline evaluation complete."

# Wait for all quantizations to finish
for pid in "${QUANT_PIDS[@]}"; do
  wait "$pid" || { echo "A quantization job failed" >&2; exit 1; }
done
echo "  All quantizations complete."

# ── Phase 2: Evaluations (GPU, sequential) + angle analyses (CPU, background)
echo ""
echo "=== Phase 2: Evaluations + angle analyses ==="

ANGLE_PIDS=()
for bit in "${BIT_WIDTHS[@]}"; do
  QUANT_OUT_DIR="${SWEEP_ROOT}/${bit}bit"

  # Launch angle analysis in background (CPU-only)
  ANGLE_DIR="${QUANT_OUT_DIR}/angle_reports"
  mkdir -p "$ANGLE_DIR"
  echo "  Starting int${bit} angle analysis (background)..."
  python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
    "$CKPT_DIR/ckpt.pt" \
    "$PATTERN" \
    --compare-ckpt "$QUANT_OUT_DIR/ckpt.pt" \
    --comparison-csv "${ANGLE_DIR}/angles.csv" \
    --angle-units degrees \
    --no-colorize &
  ANGLE_PIDS+=($!)

  # Run evaluation sequentially (GPU — avoid contention)
  EVAL_DIR="${EVAL_ROOT}/${bit}bit"
  mkdir -p "$EVAL_DIR"
  echo "  Evaluating int${bit} checkpoint..."
  python3 sample.py \
    --out_dir "$QUANT_OUT_DIR" \
    --eval_only \
    --eval_dataset "$EVAL_DATASET" \
    --eval_iters "$EVAL_ITERS" \
    --eval_output_dir "$EVAL_DIR"
done

# Wait for any remaining angle analyses
echo "  Waiting for angle analyses to finish..."
for pid in "${ANGLE_PIDS[@]}"; do
  wait "$pid" || { echo "An angle analysis job failed" >&2; exit 1; }
done
echo "  All evaluations and angle analyses complete."

# ── Phase 3: Aggregate results, write CSV, and plot ─────────────────────────
echo ""
echo "=== Phase 3: Generate summary CSV and plots ==="

python3 - \
  "$CKPT_DIR" \
  "$SWEEP_ROOT" \
  "$EVAL_ROOT" \
  "$SUMMARY_ROOT" \
  --bits "${BIT_WIDTHS[@]}" <<'PY'
import argparse
import csv
import json
import math
import os
import statistics
from typing import Dict, List, Optional

parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir")
parser.add_argument("sweep_root")
parser.add_argument("eval_root")
parser.add_argument("summary_root")
parser.add_argument("--bits", nargs="+", type=int, required=True)
args = parser.parse_args()

ckpt_dir = os.path.abspath(args.ckpt_dir)
sweep_root = os.path.abspath(args.sweep_root)
eval_root = os.path.abspath(args.eval_root)
summary_root = os.path.abspath(args.summary_root)
sweep_bits = sorted(args.bits, reverse=True)

# ── Load baseline loss ──────────────────────────────────────────────────────
baseline_eval = os.path.join(eval_root, "fp32", "eval_loss.txt")
if not os.path.exists(baseline_eval):
    raise SystemExit(f"Missing baseline evaluation at {baseline_eval}")
with open(baseline_eval, encoding="utf-8") as fh:
    baseline_loss = json.load(fh).get("val")
if baseline_loss is None:
    raise SystemExit(f"No 'val' key in {baseline_eval}")
baseline_loss = float(baseline_loss)

# ── Load per-bit-width results ──────────────────────────────────────────────
def load_angle_summary(bit: int) -> Dict[str, float]:
    angle_csv = os.path.join(sweep_root, f"{bit}bit", "angle_reports", "angles.csv")
    if not os.path.exists(angle_csv):
        return {}
    angles: List[float] = []
    cosines: List[float] = []
    with open(angle_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = float(row.get("angle", "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(a):
                angles.append(a)
            try:
                c = float(row.get("cosine_similarity", "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(c):
                cosines.append(c)
    if not angles:
        return {}
    result: Dict[str, float] = {
        "mean_angle": statistics.mean(angles),
        "median_angle": statistics.median(angles),
        "max_angle": max(angles),
        "min_angle": min(angles),
    }
    if cosines:
        result["mean_cosine"] = statistics.mean(cosines)
    return result

entries = []
for bit in sweep_bits:
    # Validation loss
    loss_path = os.path.join(eval_root, f"{bit}bit", "eval_loss.txt")
    if not os.path.exists(loss_path):
        raise SystemExit(f"Missing evaluation at {loss_path}")
    with open(loss_path, encoding="utf-8") as fh:
        val_loss = float(json.load(fh)["val"])

    angle_info = load_angle_summary(bit)

    entries.append({
        "bits": bit,
        "label": f"int{bit}",
        "val_loss": val_loss,
        "mean_angle": angle_info.get("mean_angle"),
        "median_angle": angle_info.get("median_angle"),
        "max_angle": angle_info.get("max_angle"),
        "min_angle": angle_info.get("min_angle"),
        "mean_cosine": angle_info.get("mean_cosine"),
    })

entries.sort(key=lambda e: e["bits"], reverse=True)

# ── Write CSV ───────────────────────────────────────────────────────────────
os.makedirs(summary_root, exist_ok=True)
csv_path = os.path.join(summary_root, "symmetric_vector_quantization_analysis.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csv_out:
    fieldnames = [
        "bits", "label", "val_loss",
        "mean_angle_deg", "median_angle_deg", "max_angle_deg", "min_angle_deg",
        "mean_cosine_similarity",
    ]
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    writer.writeheader()

    # Write baseline row
    writer.writerow({
        "bits": 32,
        "label": "fp32",
        "val_loss": f"{baseline_loss:.8f}",
        "mean_angle_deg": "0.0",
        "median_angle_deg": "0.0",
        "max_angle_deg": "0.0",
        "min_angle_deg": "0.0",
        "mean_cosine_similarity": "1.0",
    })

    for entry in entries:
        writer.writerow({
            "bits": entry["bits"],
            "label": entry["label"],
            "val_loss": f"{entry['val_loss']:.8f}",
            "mean_angle_deg": "" if entry["mean_angle"] is None else f"{entry['mean_angle']:.8f}",
            "median_angle_deg": "" if entry["median_angle"] is None else f"{entry['median_angle']:.8f}",
            "max_angle_deg": "" if entry["max_angle"] is None else f"{entry['max_angle']:.8f}",
            "min_angle_deg": "" if entry["min_angle"] is None else f"{entry['min_angle']:.8f}",
            "mean_cosine_similarity": "" if entry["mean_cosine"] is None else f"{entry['mean_cosine']:.8f}",
        })

print(f"Wrote summary CSV to {csv_path}")

# ── Plot ────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping plot generation.")
    raise SystemExit(0)

plt.style.use("seaborn-v0_8")
fig, (ax_loss, ax_angle) = plt.subplots(1, 2, figsize=(14, 6))

bits = [e["bits"] for e in entries]
losses = [e["val_loss"] for e in entries]
mean_angles = [e["mean_angle"] for e in entries]
median_angles = [e["median_angle"] for e in entries]

# ── Left panel: Validation loss vs. bit-width ───────────────────────────────
ax_loss.plot(bits, losses, marker="o", color="tab:blue", linewidth=2, label="Symmetric vector PTQ")
ax_loss.axhline(baseline_loss, linestyle="--", color="tab:orange", linewidth=1.5, label="fp32 baseline")
ax_loss.invert_xaxis()
ax_loss.set_xticks(bits)
ax_loss.set_xticklabels([f"int{b}" for b in bits])
ax_loss.set_xlabel("Bit-width")
ax_loss.set_ylabel("Validation loss")
ax_loss.set_title("Validation Loss vs. Bit-width\n(Symmetric Per-Vector Quantization)")
ax_loss.legend(fontsize=9)
ax_loss.grid(True, which="both", linestyle=":", linewidth=0.5)

# ── Right panel: Angular distortion vs. bit-width ───────────────────────────
valid_mean = [(b, a) for b, a in zip(bits, mean_angles) if a is not None]
valid_median = [(b, a) for b, a in zip(bits, median_angles) if a is not None]

if valid_mean:
    vb, va = zip(*valid_mean)
    ax_angle.plot(vb, va, marker="o", color="tab:red", linewidth=2, label="Mean angle")
if valid_median:
    vb, va = zip(*valid_median)
    ax_angle.plot(vb, va, marker="s", color="tab:purple", linewidth=2, label="Median angle")

ax_angle.invert_xaxis()
if bits:
    ax_angle.set_xticks(bits)
    ax_angle.set_xticklabels([f"int{b}" for b in bits])
ax_angle.set_xlabel("Bit-width")
ax_angle.set_ylabel("Angular distortion (degrees)")
ax_angle.set_title("Angular Distortion vs. Bit-width\n(Symmetric Per-Vector Quantization)")
ax_angle.legend(fontsize=9)
ax_angle.grid(True, which="both", linestyle=":", linewidth=0.5)

fig.tight_layout()
plot_path = os.path.join(summary_root, "symmetric_vector_quantization_analysis.png")
fig.savefig(plot_path, dpi=200)
print(f"Wrote plot to {plot_path}")
PY

echo ""
echo "============================================================"
echo "Analysis complete."
echo "  Summary CSV:  $SUMMARY_ROOT/symmetric_vector_quantization_analysis.csv"
echo "  Summary plot: $SUMMARY_ROOT/symmetric_vector_quantization_analysis.png"
echo "============================================================"
