#!/bin/bash
# demos/fake_ptq_vector_eval_demo_minipile.sh
#
# Runs the fake PTQ pipeline using per-vector quantization heuristics inspired by
# the JL transform initialization script. For each bit-width in the sweep the
# script quantizes the checkpoint with per-vector granularity, evaluates the
# model on minipile, and records validation loss plus angle statistics relative
# to the fp32 baseline checkpoint. It also sweeps embedding Gaussian noise
# perturbation magnitudes to compare loss/angle distortion vs alpha.

set -euo pipefail

EVAL_DATASET_DIR="data/minipile"
OUT_DIR="out_fake_ptq_minipile"
VECTOR_SWEEP_ROOT="${OUT_DIR}_vector_sweep"
SUMMARY_ROOT="${OUT_DIR}_quantization_summaries"
NOISE_SWEEP_ROOT="${OUT_DIR}_gaussian_noise_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256

BIT_START=8
BIT_STOP=3
BIT_STEP=-1

ALPHAS=(0.05 0.1 0.2 0.3)

usage() {
  cat <<'USAGE'
Usage: demos/fake_ptq_vector_eval_demo_minipile.sh [--bit-start N] [--bit-stop N] [--bit-step N]

  --bit-start  Starting bit-width for the sweep (default: 16)
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

mkdir -p "$EVAL_DATASET_DIR"

echo "=== Step 1: Prepare the minipile dataset ==="
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "$VECTOR_SWEEP_ROOT" "$SUMMARY_ROOT" "$NOISE_SWEEP_ROOT"

echo "=== Step 2: Train a reference model on minipile (if needed) ==="
if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset minipile \
    --out_dir "$OUT_DIR" \
    --n_layer 6 \
    --n_head 6 \
    --n_embd 384 \
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
    --eta_variant "iteration" \
    --compile
else
  echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate the baseline (fp32) checkpoint ==="
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --eval_only \
  --eval_dataset minipile

PATTERN='transformer\.h\.[0-9]+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight'

step=4
for bit in "${BITS[@]}"; do
  QUANT_OUT_DIR="${VECTOR_SWEEP_ROOT}/${bit}bit"
  mkdir -p "$QUANT_OUT_DIR"

  echo "=== Step ${step}: Quantize to ${bit}-bit weights (vector) ==="
  if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
    python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
      --out_dir "$QUANT_OUT_DIR" \
      --num_bits "$bit" \
      --granularity vector
  else
    echo "Found existing ${bit}-bit checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
  fi

  step=$((step + 1))

  echo "=== Step ${step}: Evaluate the ${bit}-bit checkpoint (vector) ==="
  python3 sample.py \
    --out_dir "$QUANT_OUT_DIR" \
    --eval_only \
    --eval_dataset minipile

  step=$((step + 1))

  echo "=== Step ${step}: Compare vector angles against baseline ==="
  ANGLE_DIR="${QUANT_OUT_DIR}/angle_reports"
  mkdir -p "$ANGLE_DIR"
  python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
    "$OUT_DIR/ckpt.pt" \
    "$PATTERN" \
    --compare-ckpt "$QUANT_OUT_DIR/ckpt.pt" \
    --comparison-csv "${ANGLE_DIR}/per_vector_angles.csv" \
    --angle-units degrees \
    --no-colorize

  step=$((step + 1))
done

echo "=== Step ${step}: Sweep embedding Gaussian noise perturbations ==="
python3 quantizations/ptq/embedding_gaussian_noise_ckpt.py "$OUT_DIR" \
  --out_dir "$NOISE_SWEEP_ROOT" \
  --alphas "${ALPHAS[@]}"

step=$((step + 1))

for alpha in "${ALPHAS[@]}"; do
  alpha_tag="${alpha//./p}"
  NOISE_OUT_DIR="${NOISE_SWEEP_ROOT}/alpha_${alpha_tag}"

  echo "=== Step ${step}: Evaluate noise checkpoint (alpha=${alpha}) ==="
  python3 sample.py \
    --out_dir "$NOISE_OUT_DIR" \
    --eval_only \
    --eval_dataset minipile

  step=$((step + 1))

  echo "=== Step ${step}: Compare noise angles against baseline (alpha=${alpha}) ==="
  ANGLE_DIR="${NOISE_OUT_DIR}/angle_reports"
  mkdir -p "$ANGLE_DIR"
  python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
    "$OUT_DIR/ckpt.pt" \
    "$PATTERN" \
    --compare-ckpt "$NOISE_OUT_DIR/ckpt.pt" \
    --comparison-csv "${ANGLE_DIR}/per_vector_angles.csv" \
    --angle-units degrees \
    --no-colorize

  step=$((step + 1))
done

python3 - "$OUT_DIR" "$VECTOR_SWEEP_ROOT" "$NOISE_SWEEP_ROOT" "$SUMMARY_ROOT" "${BITS[@]}" <<'PY'
import csv
import json
import math
import os
import statistics
import sys

out_dir = os.path.abspath(sys.argv[1])
vector_root = os.path.abspath(sys.argv[2])
noise_root = os.path.abspath(sys.argv[3])
summary_root = os.path.abspath(sys.argv[4])
sweep_bits = [int(arg) for arg in sys.argv[5:]]

if not sweep_bits:
    raise SystemExit("No bit-width sweep values provided to summary helper")

baseline_eval = os.path.join(out_dir, "eval_loss.txt")
if not os.path.exists(baseline_eval):
    raise SystemExit(f"Missing baseline evaluation summary at {baseline_eval}")

with open(baseline_eval, encoding="utf-8") as fh:
    baseline_data = json.load(fh)
baseline_loss = baseline_data.get("val")
if baseline_loss is None:
    raise SystemExit(f"No 'val' key found in {baseline_eval}")

def load_sweep(root: str, granularity: str) -> list[dict[str, object]]:
    if not os.path.isdir(root):
        raise SystemExit(f"Expected sweep root at {root}")

    entries: list[dict[str, object]] = []
    angle_suffix = "per_vector_angles.csv"

    for bit in sweep_bits:
        loss_path = os.path.join(root, f"{bit}bit", "eval_loss.txt")
        if not os.path.exists(loss_path):
            raise SystemExit(f"Missing evaluation summary at {loss_path}")
        with open(loss_path, encoding="utf-8") as fh:
            eval_data = json.load(fh)
        loss = eval_data.get("val")
        if loss is None:
            raise SystemExit(f"No 'val' key found in {loss_path}")

        angle_csv = os.path.join(root, f"{bit}bit", "angle_reports", angle_suffix)
        angle_summary = None
        if os.path.exists(angle_csv):
            angles: list[float] = []
            cosines: list[float] = []
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
            if angles:
                angle_summary = {
                    "mean_angle": statistics.mean(angles),
                    "median_angle": statistics.median(angles),
                    "mean_cosine": statistics.mean(cosines) if cosines else float("nan"),
                }

        entries.append(
            {
                "bits": bit,
                "granularity": granularity,
                "label": f"{bit}-bit {granularity}",
                "val_loss": float(loss),
                "mean_angle": None if angle_summary is None else angle_summary["mean_angle"],
                "median_angle": None if angle_summary is None else angle_summary["median_angle"],
                "mean_cosine": None if angle_summary is None else angle_summary["mean_cosine"],
            }
        )

    return entries


def load_noise_sweep(root: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if not os.path.isdir(root):
        raise SystemExit(f"Expected noise sweep root at {root}")

    for name in sorted(os.listdir(root)):
        if not name.startswith("alpha_"):
            continue
        alpha_str = name.split("alpha_", 1)[1].replace("p", ".")
        try:
            alpha_val = float(alpha_str)
        except ValueError:
            continue

        loss_path = os.path.join(root, name, "eval_loss.txt")
        if not os.path.exists(loss_path):
            raise SystemExit(f"Missing evaluation summary at {loss_path}")
        with open(loss_path, encoding="utf-8") as fh:
            eval_data = json.load(fh)
        loss = eval_data.get("val")
        if loss is None:
            raise SystemExit(f"No 'val' key found in {loss_path}")

        angle_csv = os.path.join(root, name, "angle_reports", "per_vector_angles.csv")
        angle_summary = None
        if os.path.exists(angle_csv):
            angles: list[float] = []
            with open(angle_csv, newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    try:
                        angle_val = float(row.get("angle", "nan"))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(angle_val):
                        angles.append(angle_val)
            if angles:
                angle_summary = {
                    "mean_angle": statistics.mean(angles),
                    "median_angle": statistics.median(angles),
                }

        entries.append(
            {
                "alpha": alpha_val,
                "label": f"alpha={alpha_val:g}",
                "val_loss": float(loss),
                "mean_angle": None if angle_summary is None else angle_summary["mean_angle"],
                "median_angle": None if angle_summary is None else angle_summary["median_angle"],
            }
        )

    entries.sort(key=lambda item: item["alpha"])
    return entries


baseline_entry = {
    "bits": 32,
    "granularity": "fp32",
    "label": "fp32 baseline",
    "val_loss": float(baseline_loss),
    "mean_angle": None,
    "median_angle": None,
    "mean_cosine": None,
}

all_entries = [baseline_entry]
all_entries.extend(load_sweep(vector_root, "vector"))
noise_entries = load_noise_sweep(noise_root)

all_entries.sort(key=lambda item: (item["granularity"] != "fp32", -item["bits"]))

csv_path = os.path.join(summary_root, "quantization_eval_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csv_out:
    fieldnames = [
        "bits",
        "granularity",
        "label",
        "val_loss",
        "mean_angle_deg",
        "median_angle_deg",
        "mean_cosine_similarity",
    ]
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    writer.writeheader()
    for entry in all_entries:
        writer.writerow(
            {
                "bits": entry["bits"],
                "granularity": entry["granularity"],
                "label": entry["label"],
                "val_loss": f"{entry['val_loss']:.8f}",
                "mean_angle_deg": "" if entry["mean_angle"] is None else f"{entry['mean_angle']:.8f}",
                "median_angle_deg": "" if entry["median_angle"] is None else f"{entry['median_angle']:.8f}",
                "mean_cosine_similarity": "" if entry["mean_cosine"] is None else f"{entry['mean_cosine']:.8f}",
            }
        )

noise_csv_path = os.path.join(summary_root, "gaussian_noise_eval_summary.csv")
with open(noise_csv_path, "w", newline="", encoding="utf-8") as csv_out:
    fieldnames = [
        "alpha",
        "label",
        "val_loss",
        "mean_angle_deg",
        "median_angle_deg",
    ]
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    writer.writeheader()
    for entry in noise_entries:
        writer.writerow(
            {
                "alpha": f"{entry['alpha']:.6f}",
                "label": entry["label"],
                "val_loss": f"{entry['val_loss']:.8f}",
                "mean_angle_deg": "" if entry["mean_angle"] is None else f"{entry['mean_angle']:.8f}",
                "median_angle_deg": "" if entry["median_angle"] is None else f"{entry['median_angle']:.8f}",
            }
        )

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting dependency issues
    raise SystemExit(f"Failed to import matplotlib for plotting: {exc}") from exc

plt.style.use("seaborn-v0_8")

fig, ((ax_loss, ax_angle), (ax_noise_loss, ax_noise_angle)) = plt.subplots(
    2, 2, figsize=(13, 9)
)

subset = [entry for entry in all_entries if entry["granularity"] == "vector"]
subset.sort(key=lambda item: item["bits"], reverse=True)
bits = [entry["bits"] for entry in subset]
losses = [entry["val_loss"] for entry in subset]
ax_loss.plot(bits, losses, marker="o", color="tab:blue", label="vector quant")

angles = [entry["mean_angle"] for entry in subset]
valid_pairs = [(b, a) for b, a in zip(bits, angles) if a is not None]
if valid_pairs:
    vb, va = zip(*valid_pairs)
    ax_angle.plot(vb, va, marker="o", color="tab:blue", label="vector quant")

ax_loss.axhline(baseline_entry["val_loss"], color="tab:green", linestyle="--", label="fp32 baseline")
ax_loss.set_xlabel("Bits")
ax_loss.set_ylabel("Validation loss")
ax_loss.set_title("Validation loss vs. bit-width (vector PTQ)")
ax_loss.legend()
ax_loss.grid(True, which="both", linestyle=":", linewidth=0.5)

ax_angle.set_xlabel("Bits")
ax_angle.set_ylabel("Mean angle (degrees)")
ax_angle.set_title("Mean angle vs. bit-width (vector PTQ)")
ax_angle.legend()
ax_angle.grid(True, which="both", linestyle=":", linewidth=0.5)

if noise_entries:
    alphas = [entry["alpha"] for entry in noise_entries]
    noise_losses = [entry["val_loss"] for entry in noise_entries]
    noise_angles = [entry["mean_angle"] for entry in noise_entries]

    ax_noise_loss.plot(alphas, noise_losses, marker="o", color="tab:purple", label="gaussian noise")
    ax_noise_loss.axhline(baseline_entry["val_loss"], color="tab:green", linestyle="--", label="fp32 baseline")
    ax_noise_loss.set_xlabel("Alpha")
    ax_noise_loss.set_ylabel("Validation loss")
    ax_noise_loss.set_title("Validation loss vs. noise alpha")
    ax_noise_loss.legend()
    ax_noise_loss.grid(True, which="both", linestyle=":", linewidth=0.5)

    valid_noise_pairs = [(a, ang) for a, ang in zip(alphas, noise_angles) if ang is not None]
    if valid_noise_pairs:
        na, nv = zip(*valid_noise_pairs)
        ax_noise_angle.plot(na, nv, marker="o", color="tab:purple", label="gaussian noise")
    ax_noise_angle.set_xlabel("Alpha")
    ax_noise_angle.set_ylabel("Mean angle (degrees)")
    ax_noise_angle.set_title("Mean angle vs. noise alpha")
    ax_noise_angle.legend()
    ax_noise_angle.grid(True, which="both", linestyle=":", linewidth=0.5)
else:
    ax_noise_loss.set_visible(False)
    ax_noise_angle.set_visible(False)

fig.tight_layout()

plot_path = os.path.join(summary_root, "quantization_eval_summary.png")
fig.savefig(plot_path, dpi=200)

print(f"Wrote summary CSV to {csv_path}")
print(f"Wrote noise summary CSV to {noise_csv_path}")
print(f"Wrote comparison plot to {plot_path}")
PY

echo "Quantization sweeps complete. Evaluation summaries live in $SUMMARY_ROOT."
