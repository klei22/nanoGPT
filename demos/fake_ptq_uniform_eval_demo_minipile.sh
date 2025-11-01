#!/bin/bash
# demos/fake_ptq_uniform_eval_demo_minipile.sh
#
# Runs fake PTQ sweeps for both baseline and PKL-parameterized minipile models.
# The script prepares the dataset, trains (or reuses) each reference model,
# quantizes checkpoints across a configurable bit-width range, evaluates every
# quantized checkpoint, and plots validation loss, angular drift, and size trends
# against the integer precision level.

set -euo pipefail

EVAL_DATASET_DIR="data/minipile"
BASE_OUT_DIR="out_fake_ptq_minipile"
PKL_OUT_DIR="out_fake_ptq_minipile_pkl"
BASE_SWEEP_ROOT="${BASE_OUT_DIR}_uniform_sweep"
PKL_SWEEP_ROOT="${PKL_OUT_DIR}_uniform_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256
PKL_SCALE="1.4142135623730951"

BIT_START=16
BIT_STOP=3
BIT_STEP=-1

usage() {
  cat <<'EOF'
Usage: demos/fake_ptq_uniform_eval_demo.sh [--bit-start N] [--bit-stop N] [--bit-step N]

  --bit-start  Starting bit-width for the sweep (default: 8)
  --bit-stop   Final bit-width for the sweep (default: 3)
  --bit-step   Step increment for the sweep (default: -1)
  --help       Show this help message and exit
EOF
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

echo "Sweeping uniform weight bit-widths: ${BITS[*]}"

mkdir -p "$EVAL_DATASET_DIR"

step=1
log_step() {
  echo "=== Step ${step}: $* ==="
  step=$((step + 1))
}

log_step "Prepare the minipile dataset"
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

log_step "Train a baseline (non-PKL) reference model on minipile (if needed)"
if [ ! -f "$BASE_OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset minipile \
    --out_dir "$BASE_OUT_DIR" \
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
  echo "Found existing checkpoint at $BASE_OUT_DIR/ckpt.pt; skipping training."
fi

log_step "Train a PKL-parameterized reference model on minipile (if needed)"
if [ ! -f "$PKL_OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset minipile \
    --out_dir "$PKL_OUT_DIR" \
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
    --compile \
    --linear_variant_attn "pkl_linear" \
    --linear_variant_mlp "pkl_linear" \
    --use_pkl_wte \
    --use_pkl_lm_head \
    --pkl_linear_scale "$PKL_SCALE" \
    --pkl_wte_scale "$PKL_SCALE" \
    --pkl_lm_head_scale "$PKL_SCALE"
else
  echo "Found existing checkpoint at $PKL_OUT_DIR/ckpt.pt; skipping training."
fi

run_sweep() {
  local label="$1"
  local reference_dir="$2"
  local sweep_root="$3"
  shift 3
  local -a bits=("$@")

  mkdir -p "$sweep_root"

  log_step "Evaluate the ${label} fp32 checkpoint"
  python3 sample.py \
    --out_dir "$reference_dir" \
    --eval_only \
    --eval_dataset minipile \
    --eval_iters "$EVAL_ITERS"

  for bit in "${bits[@]}"; do
    local quant_out_dir="${sweep_root}/${bit}bit"
    mkdir -p "$quant_out_dir"

    log_step "Quantize the ${label} checkpoint to ${bit}-bit weights"
    if [ ! -f "$quant_out_dir/ckpt.pt" ]; then
      python3 quantizations/ptq/fake_quantize_ckpt.py "$reference_dir" \
        --out_dir "$quant_out_dir" \
        --num_bits "$bit"
    else
      echo "Found existing ${bit}-bit checkpoint at $quant_out_dir/ckpt.pt; skipping quantization."
    fi

    log_step "Evaluate the ${label} ${bit}-bit checkpoint"
    python3 sample.py \
      --out_dir "$quant_out_dir" \
      --eval_only \
      --eval_dataset minipile \
      --eval_iters "$EVAL_ITERS"
  done

  log_step "Summarize ${label} quantization sweep results"
  python3 - "$reference_dir" "$sweep_root" "${bits[@]}" <<'PY'
import json
import math
import os
import sys
from typing import Dict, Iterable

import torch


def load_vector(path: str) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_obj = checkpoint["model"]
    else:
        state_obj = checkpoint

    if isinstance(state_obj, dict):
        tensors = []
        for value in state_obj.values():
            if torch.is_tensor(value):
                tensors.append(value.float().reshape(-1))
        if not tensors:
            raise RuntimeError(f"No tensors found in checkpoint: {path}")
        return torch.cat(tensors)

    to_state_dict = getattr(state_obj, "state_dict", None)
    if to_state_dict is None:
        raise RuntimeError("Checkpoint object does not provide state_dict()")
    tensor_values: Dict[str, torch.Tensor] = {
        k: v.float().reshape(-1)
        for k, v in to_state_dict().items()
        if torch.is_tensor(v)
    }
    if not tensor_values:
        raise RuntimeError(f"No tensors found in checkpoint: {path}")
    return torch.cat(list(tensor_values.values()))


def cosine_and_angle(vec_a: torch.Tensor, vec_b: torch.Tensor) -> tuple[float, float]:
    a = vec_a.double()
    b = vec_b.double()
    dot = torch.dot(a, b).item()
    denom = max(a.norm().item() * b.norm().item(), 1e-12)
    cosine = max(min(dot / denom, 1.0), -1.0)
    angle = math.degrees(math.acos(cosine))
    return cosine, angle


def levels_for_bits(bits: int) -> float:
    return float("inf") if bits <= 0 else float(2**bits)


reference_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])
sweep_bits = [int(arg) for arg in sys.argv[3:]]

entries = [("fp32", 32, os.path.join(reference_dir, "eval_loss.txt"), reference_dir)]
for bit in sweep_bits:
    quant_dir = os.path.join(sweep_root, f"{bit}bit")
    entries.append((f"{bit}-bit", bit, os.path.join(quant_dir, "eval_loss.txt"), quant_dir))

baseline_ckpt = os.path.join(reference_dir, "ckpt.pt")
if not os.path.exists(baseline_ckpt):
    raise SystemExit(f"Missing baseline checkpoint at {baseline_ckpt}")

baseline_vec = load_vector(baseline_ckpt)

results = []
baseline_loss = None

for label, bit, eval_path, ckpt_dir in entries:
    if not os.path.exists(eval_path):
        raise SystemExit(f"Missing evaluation summary at {eval_path}")
    if not os.path.exists(os.path.join(ckpt_dir, "ckpt.pt")):
        raise SystemExit(f"Missing checkpoint at {ckpt_dir}/ckpt.pt")

    with open(eval_path, encoding="utf-8") as fh:
        data = json.load(fh)
    val = data.get("val")
    if val is None:
        raise SystemExit(f"No 'val' key found in {eval_path}")

    vec = load_vector(os.path.join(ckpt_dir, "ckpt.pt"))
    if vec.numel() != baseline_vec.numel():
        raise SystemExit(
            "Checkpoint parameter size mismatch when computing angular metrics: "
            f"baseline has {baseline_vec.numel()} elements, {label} has {vec.numel()} elements"
        )
    cosine, angle = cosine_and_angle(baseline_vec, vec)
    size_bytes = os.path.getsize(os.path.join(ckpt_dir, "ckpt.pt"))
    result = {
        "bits": bit,
        "label": label,
        "val_loss": float(val),
        "cosine_similarity": float(cosine),
        "angle_degrees": float(angle),
        "checkpoint_bytes": int(size_bytes),
        "int_levels": levels_for_bits(bit),
    }
    results.append(result)
    if label.lower() == "fp32":
        baseline_loss = result["val_loss"]

if baseline_loss is None:
    raise SystemExit("Unable to find baseline loss entry")

for result in results:
    result["delta_val_loss"] = result["val_loss"] - baseline_loss
    result["checkpoint_megabytes"] = result["checkpoint_bytes"] / (1024.0 * 1024.0)

results.sort(key=lambda item: item["bits"], reverse=True)

csv_path = os.path.join(sweep_root, "uniform_quantization_eval.csv")
with open(csv_path, "w", encoding="utf-8") as csv_file:
    headers = [
        "bits",
        "label",
        "val_loss",
        "delta_val_loss",
        "cosine_similarity",
        "angle_degrees",
        "checkpoint_bytes",
        "checkpoint_megabytes",
        "int_levels",
    ]
    csv_file.write(",".join(headers) + "\n")
    for entry in results:
        row = [
            str(entry["bits"]),
            entry["label"],
            f"{entry['val_loss']:.8f}",
            f"{entry['delta_val_loss']:.8f}",
            f"{entry['cosine_similarity']:.8f}",
            f"{entry['angle_degrees']:.8f}",
            str(entry["checkpoint_bytes"]),
            f"{entry['checkpoint_megabytes']:.4f}",
            f"{entry['int_levels']:.0f}" if math.isfinite(entry["int_levels"]) else "inf",
        ]
        csv_file.write(",".join(row) + "\n")
print(f"Wrote summary CSV to {csv_path}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping plot generation.")
else:
    baseline_entry = next((entry for entry in results if entry["label"].lower() == "fp32"), None)
    quantized_entries = [entry for entry in results if entry is not baseline_entry]

    def make_plot(y_key: str, *, ylabel: str, title_suffix: str, filename: str) -> None:
        xs = [entry["bits"] for entry in quantized_entries]
        ys = [entry[y_key] for entry in quantized_entries]
        labels = [entry["label"] for entry in quantized_entries]

        plt.figure(figsize=(8, 4.5))
        line_handle = None
        if xs:
            (line_handle,) = plt.plot(xs, ys, marker="o", label="Quantized checkpoints")

        tick_bits = []
        tick_labels = []
        if baseline_entry is not None:
            tick_bits.append(baseline_entry["bits"])
            tick_labels.append(baseline_entry["label"])
        tick_bits.extend(xs)
        tick_labels.extend(labels)

        plt.gca().invert_xaxis()
        if tick_bits:
            plt.xticks(tick_bits, tick_labels, rotation=30)
        plt.xlabel("Integer level (bits)")
        plt.ylabel(ylabel)
        plt.title(f"{title_suffix} vs integer level")
        plt.grid(True, linestyle="--", alpha=0.4)

        baseline_handle = None
        legend_handles = []
        legend_labels = []
        if line_handle is not None:
            legend_handles.append(line_handle)
            legend_labels.append("Quantized checkpoints")
        if baseline_entry is not None:
            baseline_value = baseline_entry[y_key]
            baseline_handle = plt.axhline(
                baseline_value,
                linestyle="--",
                color="tab:orange",
                label="fp32 baseline",
            )
            legend_handles.append(baseline_handle)
            legend_labels.append("fp32 baseline")
        if legend_handles:
            plt.legend(legend_handles, legend_labels)

        plt.tight_layout()
        plot_path = os.path.join(sweep_root, filename)
        plt.savefig(plot_path, dpi=200)
        print(f"Wrote plot to {plot_path}")
        plt.close()

    make_plot("val_loss", ylabel="Validation loss", title_suffix="Validation loss", filename="uniform_quantization_eval.png")
    make_plot("delta_val_loss", ylabel="Δ validation loss", title_suffix="Loss delta", filename="uniform_quantization_eval_delta.png")
    make_plot("angle_degrees", ylabel="Angle (degrees)", title_suffix="Angle", filename="uniform_quantization_eval_angle.png")
    make_plot("checkpoint_megabytes", ylabel="Checkpoint size (MB)", title_suffix="Checkpoint size", filename="uniform_quantization_eval_size.png")
PY
}

run_sweep "baseline" "$BASE_OUT_DIR" "$BASE_SWEEP_ROOT" "${BITS[@]}"
run_sweep "PKL" "$PKL_OUT_DIR" "$PKL_SWEEP_ROOT" "${BITS[@]}"

echo "Sweep complete. Baseline results live in $BASE_SWEEP_ROOT; PKL results live in $PKL_SWEEP_ROOT."
