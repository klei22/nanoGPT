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
# The shell loop handles only quantization + eval.  Angular distortion is
# computed directly in the final Python aggregation step (a lightweight
# per-vector cosine similarity), avoiding the heavyweight
# checkpoint_regex_explorer.py pipeline (pairwise stats, L2 norms, histograms,
# group metrics, rich tables) that is unnecessary for these plots.

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

# ── Step 2: Evaluate the baseline (fp32) checkpoint ─────────────────────────
echo ""
echo "=== Step 2: Evaluate the baseline (fp32) checkpoint ==="
BASELINE_EVAL_DIR="${EVAL_ROOT}/fp32"
mkdir -p "$BASELINE_EVAL_DIR"
python3 sample.py \
  --out_dir "$CKPT_DIR" \
  --eval_only \
  --eval_dataset "$EVAL_DATASET" \
  --eval_iters "$EVAL_ITERS" \
  --eval_output_dir "$BASELINE_EVAL_DIR"

# ── Step 3: Quantize and evaluate each bit-width ────────────────────────────
step=3
for bit in "${BIT_WIDTHS[@]}"; do
  QUANT_OUT_DIR="${SWEEP_ROOT}/${bit}bit"
  mkdir -p "$QUANT_OUT_DIR"

  echo ""
  echo "=== Step ${step}: Quantize to int${bit} (symmetric, per-vector) ==="
  if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
    python3 quantizations/ptq/fake_quantize_ckpt.py "$CKPT_DIR" \
      --out_dir "$QUANT_OUT_DIR" \
      --num_bits "$bit" \
      --quantization symmetric \
      --granularity vector
  else
    echo "Found existing int${bit} checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping."
  fi
  step=$((step + 1))

  echo "=== Step ${step}: Evaluate int${bit} checkpoint ==="
  EVAL_DIR="${EVAL_ROOT}/${bit}bit"
  mkdir -p "$EVAL_DIR"
  python3 sample.py \
    --out_dir "$QUANT_OUT_DIR" \
    --eval_only \
    --eval_dataset "$EVAL_DATASET" \
    --eval_iters "$EVAL_ITERS" \
    --eval_output_dir "$EVAL_DIR"
  step=$((step + 1))
done

# ── Final step: Compute angles inline, aggregate CSV, and plot ──────────────
# Angular distortion is computed here directly via per-vector cosine similarity
# between the baseline and each quantized checkpoint.  This replaces the
# checkpoint_regex_explorer.py calls which also computed expensive pairwise
# intra-tensor metrics, L2 norm stats, group statistics, and histograms that
# are not needed for the summary plots.
echo ""
echo "=== Step ${step}: Generate summary CSV and plots ==="

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
import re
import statistics
from typing import Dict, List, Tuple

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

import torch
import torch.nn.functional as F

WEIGHT_PATTERN = re.compile(
    r"transformer\.h\.[0-9]+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight"
)

# ── Helpers ─────────────────────────────────────────────────────────────────
def load_state_dict(ckpt_path: str) -> Tuple[dict, int]:
    """Load state dict and return (state_dict, embedding_dim)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    # Strip compiler prefix if present
    cleaned = {}
    for k, v in sd.items():
        cleaned[k.removeprefix("_orig_mod.")] = v
    n_embd = ckpt.get("model_args", {}).get("n_embd")
    return cleaned, n_embd

def compute_angle_stats(
    baseline_sd: dict, quantized_sd: dict, embedding_dim: int
) -> Dict[str, float]:
    """Compute per-vector angular distortion between two state dicts.

    Only compares weight tensors matching WEIGHT_PATTERN along axes whose
    size equals embedding_dim — the same logic as iter_vector_views in
    checkpoint_regex_explorer.py.
    """
    all_angles: List[float] = []
    all_cosines: List[float] = []

    for name in baseline_sd:
        if not WEIGHT_PATTERN.search(name):
            continue
        base_t = baseline_sd[name].detach().float()
        quant_t = quantized_sd.get(name)
        if quant_t is None or quant_t.shape != base_t.shape:
            continue
        quant_t = quant_t.detach().float()

        for axis, axis_size in enumerate(base_t.shape):
            if axis_size != embedding_dim:
                continue
            base_vecs = base_t.movedim(axis, -1).reshape(-1, embedding_dim)
            quant_vecs = quant_t.movedim(axis, -1).reshape(-1, embedding_dim)
            if base_vecs.numel() == 0:
                continue

            cos = F.cosine_similarity(base_vecs, quant_vecs, dim=-1, eps=1e-8)
            cos = cos.clamp(-1.0, 1.0)
            angles_deg = torch.rad2deg(torch.acos(cos))

            all_angles.extend(angles_deg.tolist())
            all_cosines.extend(cos.tolist())

    if not all_angles:
        return {}
    return {
        "mean_angle": statistics.mean(all_angles),
        "median_angle": statistics.median(all_angles),
        "max_angle": max(all_angles),
        "min_angle": min(all_angles),
        "mean_cosine": statistics.mean(all_cosines) if all_cosines else float("nan"),
    }

# ── Load baseline ───────────────────────────────────────────────────────────
baseline_eval = os.path.join(eval_root, "fp32", "eval_loss.txt")
if not os.path.exists(baseline_eval):
    raise SystemExit(f"Missing baseline evaluation at {baseline_eval}")
with open(baseline_eval, encoding="utf-8") as fh:
    baseline_loss = json.load(fh).get("val")
if baseline_loss is None:
    raise SystemExit(f"No 'val' key in {baseline_eval}")
baseline_loss = float(baseline_loss)

# Load baseline state dict once for angle comparisons
print("Loading baseline checkpoint for angle comparisons...")
baseline_sd, embedding_dim = load_state_dict(os.path.join(ckpt_dir, "ckpt.pt"))
if embedding_dim is None:
    print("Warning: could not determine embedding_dim; angle analysis will be skipped.")

# ── Collect per-bit-width results ───────────────────────────────────────────
entries = []
for bit in sweep_bits:
    # Validation loss
    loss_path = os.path.join(eval_root, f"{bit}bit", "eval_loss.txt")
    if not os.path.exists(loss_path):
        raise SystemExit(f"Missing evaluation at {loss_path}")
    with open(loss_path, encoding="utf-8") as fh:
        val_loss = float(json.load(fh)["val"])

    # Angle analysis — load quantized ckpt, compare, then free it
    angle_info: Dict[str, float] = {}
    if embedding_dim is not None:
        quant_ckpt_path = os.path.join(sweep_root, f"{bit}bit", "ckpt.pt")
        if os.path.exists(quant_ckpt_path):
            print(f"  Computing angles for int{bit}...")
            quant_sd, _ = load_state_dict(quant_ckpt_path)
            angle_info = compute_angle_stats(baseline_sd, quant_sd, embedding_dim)
            del quant_sd  # free memory before loading next

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

del baseline_sd  # free baseline after all comparisons

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
