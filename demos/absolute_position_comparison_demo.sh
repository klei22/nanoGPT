#!/bin/bash
# demos/absolute_position_comparison_demo.sh
#
# Compares three position embedding strategies:
#   1. Standard absolute position embeddings (default)
#   2. RoPE (rotary position embeddings)
#   3. Multi-channel cyclic absolute position embeddings
#
# For each variant the script:
#   - Trains a model on shakespeare_char (or minipile)
#   - Samples at the training block size
#   - Samples at 2x the training block size (length extrapolation)
#   - Runs fake PTQ from int8 down to int3
#   - Plots per-iteration speed, parameter count, and quantizability

set -euo pipefail

# ── Defaults (overridable via CLI) ──────────────────────────────────────────
DATASET="shakespeare_char"
DATASET_DIR="data/shakespeare_char"
BLOCK_SIZE=256
MAX_ITERS=2000
EVAL_ITERS=200
BATCH_SIZE=64
N_LAYER=6
N_HEAD=6
N_EMBD=384
BIT_START=8
BIT_STOP=3
BIT_STEP=-1
ROOT_OUT="out_abs_pos_comparison"
EXTRA_BLOCK_SIZE=512   # 2x for length extrapolation test

usage() {
  cat <<'EOF'
Usage: demos/absolute_position_comparison_demo.sh [OPTIONS]

  --dataset         Dataset name (default: shakespeare_char)
  --block_size      Training block size (default: 256)
  --max_iters       Training iterations (default: 2000)
  --bit-start       PTQ sweep start (default: 8)
  --bit-stop        PTQ sweep stop  (default: 3)
  --bit-step        PTQ sweep step  (default: -1)
  --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)      DATASET="$2"; shift 2 ;;
    --block_size)   BLOCK_SIZE="$2"; shift 2 ;;
    --max_iters)    MAX_ITERS="$2"; shift 2 ;;
    --bit-start)    BIT_START="$2"; shift 2 ;;
    --bit-stop)     BIT_STOP="$2"; shift 2 ;;
    --bit-step)     BIT_STEP="$2"; shift 2 ;;
    --help)         usage; exit 0 ;;
    *)              echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# Resolve dataset directory
if [ "$DATASET" = "minipile" ]; then
  DATASET_DIR="data/minipile"
else
  DATASET_DIR="data/${DATASET}"
fi

EXTRA_BLOCK_SIZE=$((BLOCK_SIZE * 2))

# Generate bit-width array
if ! mapfile -t BITS < <(seq "$BIT_START" "$BIT_STEP" "$BIT_STOP"); then
  echo "Failed to generate bit-width sweep" >&2; exit 1
fi
if [ "${#BITS[@]}" -eq 0 ]; then
  echo "Bit-width sweep is empty" >&2; exit 1
fi

echo "=== Absolute Position Embedding Comparison Demo ==="
echo "Dataset: $DATASET | Block size: $BLOCK_SIZE | Iters: $MAX_ITERS"
echo "PTQ sweep: ${BITS[*]}"
echo ""

# ── Step 1: Prepare dataset ────────────────────────────────────────────────
echo "=== Preparing dataset ==="
mkdir -p "$DATASET_DIR"
pushd "$DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ]; then
  if [ -f "get_dataset.sh" ]; then
    bash get_dataset.sh
  elif [ -f "prepare.py" ]; then
    python3 prepare.py
  else
    echo "No dataset preparation script found in $DATASET_DIR" >&2; exit 1
  fi
else
  echo "Dataset already prepared."
fi
popd > /dev/null

mkdir -p "$ROOT_OUT"

# ── Common training args ───────────────────────────────────────────────────
COMMON_TRAIN_ARGS=(
  --dataset "$DATASET"
  --n_layer "$N_LAYER"
  --n_head "$N_HEAD"
  --n_embd "$N_EMBD"
  --block_size "$BLOCK_SIZE"
  --batch_size "$BATCH_SIZE"
  --max_iters "$MAX_ITERS"
  --eval_iters "$EVAL_ITERS"
  --eval_interval "$MAX_ITERS"
  --use_qk_norm
  --use_qk_norm_scale
  --compile
)

# ── Define variants ────────────────────────────────────────────────────────
declare -a VARIANT_NAMES=("standard_abs" "rope" "multi_channel_cyclic")
declare -a VARIANT_LABELS=("Standard Abs Pos" "RoPE" "Multi-Channel Cyclic")

# ── Train each variant ─────────────────────────────────────────────────────
for i in "${!VARIANT_NAMES[@]}"; do
  VNAME="${VARIANT_NAMES[$i]}"
  VLABEL="${VARIANT_LABELS[$i]}"
  OUT_DIR="${ROOT_OUT}/${VNAME}"

  echo ""
  echo "=================================================================="
  echo "=== Training: $VLABEL ==="
  echo "=================================================================="

  if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
    VARIANT_ARGS=()
    case "$VNAME" in
      standard_abs)
        VARIANT_ARGS=(
          --use_abs_pos_embeddings
          --abs_pos_variant standard
          --no-use_rotary_embeddings
        )
        ;;
      rope)
        VARIANT_ARGS=(
          --no-use_abs_pos_embeddings
          --use_rotary_embeddings
        )
        ;;
      multi_channel_cyclic)
        VARIANT_ARGS=(
          --use_abs_pos_embeddings
          --abs_pos_variant multi_channel_cyclic
          --abs_pos_cycle_lengths 3 5 7 11 13 17
          --abs_pos_random_start
          --no-use_rotary_embeddings
        )
        ;;
    esac

    python3 train.py \
      "${COMMON_TRAIN_ARGS[@]}" \
      "${VARIANT_ARGS[@]}" \
      --out_dir "$OUT_DIR"
  else
    echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
  fi
done

# ── Evaluate and sample each variant ───────────────────────────────────────
for i in "${!VARIANT_NAMES[@]}"; do
  VNAME="${VARIANT_NAMES[$i]}"
  VLABEL="${VARIANT_LABELS[$i]}"
  OUT_DIR="${ROOT_OUT}/${VNAME}"

  echo ""
  echo "=== Evaluating: $VLABEL ==="

  # Eval at training block size
  python3 sample.py \
    --out_dir "$OUT_DIR" \
    --eval_only \
    --eval_dataset "$DATASET"

  # Sample at training block size
  echo "--- Sampling at block_size=$BLOCK_SIZE ---"
  python3 sample.py \
    --out_dir "$OUT_DIR" \
    --max_new_tokens "$BLOCK_SIZE" \
    --top_k 1 \
    --num_samples 2

  # Length extrapolation: sample at 2x block size
  echo "--- Length extrapolation: sampling at block_size=$EXTRA_BLOCK_SIZE ---"
  python3 sample.py \
    --out_dir "$OUT_DIR" \
    --block_size "$EXTRA_BLOCK_SIZE" \
    --max_new_tokens "$EXTRA_BLOCK_SIZE" \
    --top_k 1 \
    --num_samples 2
done

# ── PTQ sweep for each variant ─────────────────────────────────────────────
for i in "${!VARIANT_NAMES[@]}"; do
  VNAME="${VARIANT_NAMES[$i]}"
  VLABEL="${VARIANT_LABELS[$i]}"
  OUT_DIR="${ROOT_OUT}/${VNAME}"
  SWEEP_DIR="${OUT_DIR}_ptq_sweep"
  mkdir -p "$SWEEP_DIR"

  echo ""
  echo "=== PTQ sweep: $VLABEL ==="

  for bit in "${BITS[@]}"; do
    QUANT_DIR="${SWEEP_DIR}/${bit}bit"
    mkdir -p "$QUANT_DIR"

    if [ ! -f "$QUANT_DIR/ckpt.pt" ]; then
      echo "  Quantizing to ${bit}-bit..."
      python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
        --out_dir "$QUANT_DIR" \
        --num_bits "$bit"
    fi

    echo "  Evaluating ${bit}-bit..."
    python3 sample.py \
      --out_dir "$QUANT_DIR" \
      --eval_only \
      --eval_dataset "$DATASET"
  done
done

# ── Generate charts ────────────────────────────────────────────────────────
echo ""
echo "=== Generating comparison charts ==="

python3 - "$ROOT_OUT" "${BITS[*]}" <<'PYEOF'
import json
import os
import sys
import time

root_out = os.path.abspath(sys.argv[1])
bits = [int(b) for b in sys.argv[2].split()]

variant_names = ["standard_abs", "rope", "multi_channel_cyclic"]
variant_labels = ["Standard Abs Pos", "RoPE", "Multi-Channel Cyclic"]
variant_colors = ["tab:blue", "tab:orange", "tab:green"]
variant_markers = ["o", "s", "^"]

# ── Collect results ──────────────────────────────────────────────────
results = {}  # variant -> { "fp32_loss": float, "bits": {bit: loss}, "params": int, "iter_time": float }

for vname, vlabel in zip(variant_names, variant_labels):
    out_dir = os.path.join(root_out, vname)
    sweep_dir = f"{out_dir}_ptq_sweep"

    entry = {"label": vlabel, "bits": {}}

    # fp32 eval loss
    fp32_loss_path = os.path.join(out_dir, "eval_loss.txt")
    if os.path.exists(fp32_loss_path):
        with open(fp32_loss_path) as f:
            data = json.load(f)
        entry["fp32_loss"] = float(data.get("val", float("nan")))
    else:
        entry["fp32_loss"] = float("nan")

    # Parameter count from checkpoint
    try:
        import torch
        ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
        model_args = ckpt.get("model_args", {})
        # Count params from state dict
        state_dict = ckpt.get("model", {})
        total_params = sum(v.numel() for v in state_dict.values())
        entry["params"] = total_params

        # Extract iter time if available
        iter_time = ckpt.get("iter_time", None)
        if iter_time is not None:
            entry["iter_time"] = float(iter_time)
        else:
            # Try to get from config
            entry["iter_time"] = float("nan")
    except Exception as e:
        print(f"Warning: could not load checkpoint for {vname}: {e}")
        entry["params"] = 0
        entry["iter_time"] = float("nan")

    # PTQ losses
    for bit in bits:
        loss_path = os.path.join(sweep_dir, f"{bit}bit", "eval_loss.txt")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                data = json.load(f)
            entry["bits"][bit] = float(data.get("val", float("nan")))
        else:
            entry["bits"][bit] = float("nan")

    results[vname] = entry

# ── Write CSV summary ────────────────────────────────────────────────
csv_path = os.path.join(root_out, "comparison_summary.csv")
with open(csv_path, "w") as f:
    f.write("variant,label,params,fp32_loss,iter_time_ms")
    for bit in sorted(bits, reverse=True):
        f.write(f",{bit}bit_loss")
    f.write("\n")
    for vname in variant_names:
        e = results[vname]
        iter_ms = f"{e['iter_time']*1000:.1f}" if e.get("iter_time") and e["iter_time"] == e["iter_time"] else "N/A"
        f.write(f"{vname},{e['label']},{e['params']},{e['fp32_loss']:.6f},{iter_ms}")
        for bit in sorted(bits, reverse=True):
            f.write(f",{e['bits'].get(bit, float('nan')):.6f}")
        f.write("\n")
print(f"Wrote CSV summary to {csv_path}")

# ── Plot charts ──────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; skipping chart generation.")
    sys.exit(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Chart 1: PTQ Quantizability (val loss vs bit-width) ---
ax = axes[0]
sorted_bits = sorted(bits, reverse=True)
for vname, vlabel, color, marker in zip(variant_names, variant_labels, variant_colors, variant_markers):
    e = results[vname]
    y_vals = [e["bits"].get(b, float("nan")) for b in sorted_bits]
    ax.plot(sorted_bits, y_vals, marker=marker, color=color, label=vlabel)
    # fp32 baseline as dashed line
    ax.axhline(e["fp32_loss"], linestyle="--", color=color, alpha=0.4)
ax.set_xlabel("Bit-width")
ax.set_ylabel("Validation Loss")
ax.set_title("Quantizability (PTQ Val Loss vs Bit-Width)")
ax.invert_xaxis()
ax.set_xticks(sorted_bits)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# --- Chart 2: Parameter Count ---
ax = axes[1]
param_counts = [results[vn]["params"] for vn in variant_names]
bars = ax.bar(variant_labels, param_counts, color=variant_colors)
ax.set_ylabel("Total Parameters")
ax.set_title("Parameter Count Comparison")
ax.grid(True, linestyle="--", alpha=0.4, axis="y")
for bar, cnt in zip(bars, param_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{cnt/1e6:.2f}M", ha="center", va="bottom", fontsize=8)

# --- Chart 3: Per-Iteration Speed ---
ax = axes[2]
iter_times = []
for vn in variant_names:
    t = results[vn].get("iter_time", float("nan"))
    iter_times.append(t * 1000 if t == t else 0)  # ms
bars = ax.bar(variant_labels, iter_times, color=variant_colors)
ax.set_ylabel("Iteration Time (ms)")
ax.set_title("Per-Iteration Speed")
ax.grid(True, linestyle="--", alpha=0.4, axis="y")
for bar, ms in zip(bars, iter_times):
    if ms > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{ms:.1f}ms", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plot_path = os.path.join(root_out, "absolute_position_comparison.png")
plt.savefig(plot_path, dpi=200)
print(f"Wrote comparison chart to {plot_path}")

PYEOF

echo ""
echo "=== Demo complete ==="
echo "Results are in: $ROOT_OUT/"
echo "  - comparison_summary.csv"
echo "  - absolute_position_comparison.png"
