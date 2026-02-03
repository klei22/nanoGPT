#!/bin/bash
# demos/fake_ptq_vector_eval_demo_shakespeare_char_trained_noise.sh
#
# Trains three shakespeare_char checkpoints with embedding noise scales
# (0.01, 0.05, 0.10) using infinite attention, then runs per-vector/per-tensor
# fake PTQ sweeps and evaluates validation loss/angle deltas across noise
# scales (0.00, 0.01, 0.025, 0.05, 0.10).

set -euo pipefail

DATASET="shakespeare_char"
EVAL_DATASET_DIR="data/${DATASET}"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256

BIT_START=8
BIT_STOP=3
BIT_STEP=-1
TRAIN_NOISE_LEVELS=(0.00 0.01 0.05 0.10 0.15 0.20)
EVAL_NOISE_LEVELS=(0.00 0.01 0.025 0.05 0.10 0.15 0.20)

usage() {
  cat <<'USAGE'
Usage: demos/fake_ptq_vector_eval_demo_shakespeare_char_trained_noise.sh [--bit-start N] [--bit-stop N] [--bit-step N]

  --bit-start  Starting bit-width for the sweep (default: 8)
  --bit-stop   Final bit-width for the sweep (default: 3)
  --bit-step   Step increment for the sweep (default: -1)
  --help       Show this help message and exit
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

echo "Sweeping weight bit-widths: ${BITS[*]}"

echo "Training noise scales: ${TRAIN_NOISE_LEVELS[*]}"

echo "Evaluation noise scales: ${EVAL_NOISE_LEVELS[*]}"

mkdir -p "$EVAL_DATASET_DIR"

noise_tag() {
  local noise="$1"
  local formatted
  formatted=$(printf "%.3f" "$noise")
  formatted=${formatted/./p}
  echo "noise_${formatted}"
}

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

PATTERN='transformer\.h\.[0-9]+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight'

for train_noise in "${TRAIN_NOISE_LEVELS[@]}"; do
  TRAIN_TAG=$(noise_tag "$train_noise")
  OUT_DIR="out_fake_ptq_${DATASET}_${TRAIN_TAG}"
  VECTOR_SWEEP_ROOT="${OUT_DIR}_vector_sweep"
  TENSOR_SWEEP_ROOT="${OUT_DIR}_tensor_sweep"
  EVAL_ROOT="${OUT_DIR}_evals"
  SUMMARY_ROOT="${OUT_DIR}_quantization_summaries"

  mkdir -p "$VECTOR_SWEEP_ROOT" "$TENSOR_SWEEP_ROOT" "$SUMMARY_ROOT" "$EVAL_ROOT"

  echo "=== Step 2: Train baseline checkpoint (noise=${train_noise}) ==="
  if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
    python3 train.py \
      --dataset "$DATASET" \
      --out_dir "$OUT_DIR" \
      --n_layer 6 \
      --n_head 3 \
      --n_embd 384 \
      --block_size "$BLOCK_SIZE" \
      --batch_size "$BATCH_SIZE" \
      --max_iters 8000 \
      --eval_interval 1000 \
      --eval_iters "$EVAL_ITERS" \
      --learning_rate 1e-3 \
      --norm_variant_wte hyperspherenorm \
      --attention_variant infinite \
      --use_concat_heads \
      --n_qk_head_dim 100 \
      --n_v_head_dim 100 \
      --embedding_gaussian_noise_std "$train_noise" \
      --compile
  else
    echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
  fi

  echo "=== Step 3: Evaluate baseline (fp32) checkpoint ==="
  BASELINE_EVAL_DIR="${EVAL_ROOT}/fp32/$(noise_tag 0.0)"
  mkdir -p "$BASELINE_EVAL_DIR"
  python3 sample.py \
    --out_dir "$OUT_DIR" \
    --eval_only \
    --eval_dataset "$DATASET" \
    --eval_iters "$EVAL_ITERS" \
    --eval_output_dir "$BASELINE_EVAL_DIR"

  step=4
  for bit in "${BITS[@]}"; do
    for granularity in vector tensor; do
      case "$granularity" in
        vector)
          SWEEP_ROOT="$VECTOR_SWEEP_ROOT"
          ANGLE_LABEL="per_vector"
          ;;
        tensor)
          SWEEP_ROOT="$TENSOR_SWEEP_ROOT"
          ANGLE_LABEL="per_tensor"
          ;;
      esac

      QUANT_OUT_DIR="${SWEEP_ROOT}/${bit}bit"
      mkdir -p "$QUANT_OUT_DIR"

      echo "=== Step ${step}: Quantize to ${bit}-bit weights (${granularity}) ==="
      if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
        if [ "$granularity" = "vector" ]; then
          python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
            --out_dir "$QUANT_OUT_DIR" \
            --num_bits "$bit" \
            --granularity vector
        else
          python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
            --out_dir "$QUANT_OUT_DIR" \
            --num_bits "$bit"
        fi
      else
        echo "Found existing ${bit}-bit checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
      fi

      step=$((step + 1))

      for noise in "${EVAL_NOISE_LEVELS[@]}"; do
        NOISE_TAG=$(noise_tag "$noise")
        EVAL_DIR="${EVAL_ROOT}/${granularity}/${NOISE_TAG}/${bit}bit"
        mkdir -p "$EVAL_DIR"

        echo "=== Step ${step}: Evaluate ${bit}-bit checkpoint (${granularity}, noise=${noise}) ==="
        python3 sample.py \
          --out_dir "$QUANT_OUT_DIR" \
          --eval_only \
          --eval_dataset "$DATASET" \
          --eval_iters "$EVAL_ITERS" \
          --embedding_gaussian_noise_std "$noise" \
          --eval_output_dir "$EVAL_DIR"

        step=$((step + 1))
      done

      echo "=== Step ${step}: Compare ${granularity} angles against baseline ==="
      ANGLE_DIR="${QUANT_OUT_DIR}/angle_reports"
      mkdir -p "$ANGLE_DIR"
      python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
        "$OUT_DIR/ckpt.pt" \
        "$PATTERN" \
        --compare-ckpt "$QUANT_OUT_DIR/ckpt.pt" \
        --comparison-csv "${ANGLE_DIR}/${ANGLE_LABEL}_angles.csv" \
        --angle-units degrees \
        --no-colorize

      step=$((step + 1))
    done
  done

  python3 - \
    "$OUT_DIR" \
    "$VECTOR_SWEEP_ROOT" \
    "$TENSOR_SWEEP_ROOT" \
    "$EVAL_ROOT" \
    "$SUMMARY_ROOT" \
    --bits "${BITS[@]}" \
    --noise-levels "${EVAL_NOISE_LEVELS[@]}" <<'PY'
import argparse
import csv
import json
import math
import os
import statistics
from typing import Dict, List

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("vector_root")
parser.add_argument("tensor_root")
parser.add_argument("eval_root")
parser.add_argument("summary_root")
parser.add_argument("--bits", nargs="+", type=int, required=True)
parser.add_argument("--noise-levels", nargs="+", type=float, required=True)
args = parser.parse_args()

vector_root = os.path.abspath(args.vector_root)
tensor_root = os.path.abspath(args.tensor_root)
eval_root = os.path.abspath(args.eval_root)
summary_root = os.path.abspath(args.summary_root)
sweep_bits = list(args.bits)
noise_levels = list(args.noise_levels)

if not sweep_bits:
    raise SystemExit("No bit-width sweep values provided to summary helper")

if not noise_levels:
    raise SystemExit("No noise levels provided to summary helper")

def noise_tag(noise: float) -> str:
    return f"noise_{noise:.3f}".replace(".", "p")

baseline_eval = os.path.join(eval_root, "fp32", noise_tag(0.0), "eval_loss.txt")
if not os.path.exists(baseline_eval):
    raise SystemExit(f"Missing baseline evaluation summary at {baseline_eval}")

with open(baseline_eval, encoding="utf-8") as fh:
    baseline_data = json.load(fh)
baseline_loss = baseline_data.get("val")
if baseline_loss is None:
    raise SystemExit(f"No 'val' key found in {baseline_eval}")

PATTERN_SUFFIX = {
    "vector": "per_vector_angles.csv",
    "tensor": "per_tensor_angles.csv",
}

def load_angle_summary(root: str, granularity: str, bit: int) -> Dict[str, float] | None:
    angle_csv = os.path.join(root, f"{bit}bit", "angle_reports", PATTERN_SUFFIX[granularity])
    if not os.path.exists(angle_csv):
        return None

    angles: List[float] = []
    cosines: List[float] = []
    with open(angle_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                angle_val = float(row.get("angle", "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(angle_val):
                angles.append(angle_val)
            cosine_raw = row.get("cosine_similarity")
            if cosine_raw is not None:
                try:
                    cosine_val = float(cosine_raw)
                except (TypeError, ValueError):
                    cosine_val = math.nan
                if math.isfinite(cosine_val):
                    cosines.append(cosine_val)
    if not angles:
        return None
    return {
        "mean_angle": statistics.mean(angles),
        "median_angle": statistics.median(angles),
        "mean_cosine": statistics.mean(cosines) if cosines else float("nan"),
    }

angle_cache: Dict[tuple[str, int], Dict[str, float] | None] = {}

def get_angle_summary(granularity: str, bit: int) -> Dict[str, float] | None:
    key = (granularity, bit)
    if key not in angle_cache:
        root = vector_root if granularity == "vector" else tensor_root
        angle_cache[key] = load_angle_summary(root, granularity, bit)
    return angle_cache[key]

entries = []
for granularity in ("vector", "tensor"):
    for noise in noise_levels:
        tag = noise_tag(noise)
        for bit in sweep_bits:
            loss_path = os.path.join(eval_root, granularity, tag, f"{bit}bit", "eval_loss.txt")
            if not os.path.exists(loss_path):
                raise SystemExit(f"Missing evaluation summary at {loss_path}")
            with open(loss_path, encoding="utf-8") as fh:
                eval_data = json.load(fh)
            loss = eval_data.get("val")
            if loss is None:
                raise SystemExit(f"No 'val' key found in {loss_path}")

            angle_summary = get_angle_summary(granularity, bit)
            entries.append(
                {
                    "bits": bit,
                    "granularity": granularity,
                    "noise_std": noise,
                    "val_loss": float(loss),
                    "mean_angle": None if angle_summary is None else angle_summary["mean_angle"],
                    "median_angle": None if angle_summary is None else angle_summary["median_angle"],
                    "mean_cosine": None if angle_summary is None else angle_summary["mean_cosine"],
                }
            )

entries.sort(key=lambda item: (item["granularity"], item["noise_std"], -item["bits"]))

csv_path = os.path.join(summary_root, "quantization_eval_summary.csv")
os.makedirs(summary_root, exist_ok=True)
with open(csv_path, "w", newline="", encoding="utf-8") as csv_out:
    fieldnames = [
        "bits",
        "granularity",
        "noise_std",
        "val_loss",
        "mean_angle_deg",
        "median_angle_deg",
        "mean_cosine_similarity",
    ]
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    writer.writeheader()
    for entry in entries:
        writer.writerow(
            {
                "bits": entry["bits"],
                "granularity": entry["granularity"],
                "noise_std": f"{entry['noise_std']:.3f}",
                "val_loss": f"{entry['val_loss']:.8f}",
                "mean_angle_deg": "" if entry["mean_angle"] is None else f"{entry['mean_angle']:.8f}",
                "median_angle_deg": "" if entry["median_angle"] is None else f"{entry['median_angle']:.8f}",
                "mean_cosine_similarity": "" if entry["mean_cosine"] is None else f"{entry['mean_cosine']:.8f}",
            }
        )

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting dependency issues
    raise SystemExit(f"Failed to import matplotlib for plotting: {exc}") from exc

plt.style.use("seaborn-v0_8")

fig, (ax_loss, ax_angle) = plt.subplots(1, 2, figsize=(14, 6))

noise_levels_sorted = sorted(noise_levels)
noise_colors = plt.cm.viridis(
    [i / max(len(noise_levels_sorted) - 1, 1) for i in range(len(noise_levels_sorted))]
)
noise_color_map = {noise: noise_colors[i] for i, noise in enumerate(noise_levels_sorted)}
markers = {"vector": "o", "tensor": "s"}

for granularity in ("vector", "tensor"):
    subset = [entry for entry in entries if entry["granularity"] == granularity]
    for noise in noise_levels_sorted:
        noise_subset = [entry for entry in subset if entry["noise_std"] == noise]
        if not noise_subset:
            continue
        noise_subset.sort(key=lambda item: item["bits"], reverse=True)
        bits = [entry["bits"] for entry in noise_subset]
        losses = [entry["val_loss"] for entry in noise_subset]
        label = f"{granularity} noise={noise:.3f}"
        ax_loss.plot(
            bits,
            losses,
            marker=markers[granularity],
            color=noise_color_map[noise],
            label=label,
        )

        angles = [entry["mean_angle"] for entry in noise_subset]
        valid_pairs = [(b, a) for b, a in zip(bits, angles) if a is not None]
        if valid_pairs:
            vb, va = zip(*valid_pairs)
            ax_angle.plot(
                vb,
                va,
                marker=markers[granularity],
                color=noise_color_map[noise],
                label=label,
            )

ax_loss.axhline(float(baseline_loss), color="tab:green", linestyle="--", label="fp32 baseline")
ax_loss.set_xlabel("Bits")
ax_loss.set_ylabel("Validation loss")
ax_loss.set_title("Validation loss vs. bit-width")
ax_loss.legend(fontsize=8)
ax_loss.grid(True, which="both", linestyle=":", linewidth=0.5)

ax_angle.set_xlabel("Bits")
ax_angle.set_ylabel("Mean angle (degrees)")
ax_angle.set_title("Mean angle vs. bit-width")
ax_angle.legend(fontsize=8)
ax_angle.grid(True, which="both", linestyle=":", linewidth=0.5)

fig.tight_layout()

plot_path = os.path.join(summary_root, "quantization_eval_summary.png")
fig.savefig(plot_path, dpi=200)

print(f"Wrote summary CSV to {csv_path}")
print(f"Wrote comparison plot to {plot_path}")
PY

echo "Quantization sweeps complete for training noise=${train_noise}."
done
