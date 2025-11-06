#!/bin/bash
# demos/fake_ptq_float_eval_demo_minipile.sh
#
# Extends the uniform fake PTQ demo by also sweeping floating-point fake
# quantization configurations over exponent bits (4 -> 1) and mantissa bits
# (4 -> 2) for the minipile dataset. The script prepares the dataset, trains a
# reference model (if needed), evaluates the fp32 baseline, sweeps uniform
# bit-widths, and then evaluates floating-point configurations.

set -euo pipefail

EVAL_DATASET_DIR="data/minipile"
OUT_DIR="out_fake_ptq_minipile"
UNIFORM_SWEEP_ROOT="${OUT_DIR}_uniform_sweep"
FLOAT_SWEEP_ROOT="${OUT_DIR}_float_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256

BIT_START=16
BIT_STOP=3
BIT_STEP=-1
EXP_BITS=(4 3 2 1)
MAN_BITS=(4 3 2)

usage() {
  cat <<'EOH'
Usage: demos/fake_ptq_float_eval_demo_minipile.sh [options]

Options:
  --bit-start N    Starting bit-width for the uniform sweep (default: 16)
  --bit-stop N     Final bit-width for the uniform sweep (default: 3)
  --bit-step N     Step increment for the uniform sweep (default: -1)
  --exp-bits LIST  Comma-separated exponent bit list for floating sweep (default: 4,3,2,1)
  --man-bits LIST  Comma-separated mantissa bit list for floating sweep (default: 4,3,2)
  --help           Show this help message and exit

The floating-point sweep evaluates every combination of exponent and mantissa
bit settings using fake PTQ with the new floating-point option.
EOH
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
    --exp-bits)
      IFS=',' read -ra EXP_BITS <<< "$(echo "$2" | tr ' ' ',')"
      shift 2
      ;;
    --man-bits)
      IFS=',' read -ra MAN_BITS <<< "$(echo "$2" | tr ' ' ',')"
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

if [ "${#EXP_BITS[@]}" -eq 0 ] || [ "${#MAN_BITS[@]}" -eq 0 ]; then
  echo "Floating-point sweep lists cannot be empty" >&2
  exit 1
fi

for exp in "${EXP_BITS[@]}"; do
  if ! [[ "$exp" =~ ^-?[0-9]+$ ]]; then
    echo "Invalid exponent bit entry: $exp" >&2
    exit 1
  fi
  if (( exp <= 0 )); then
    echo "Exponent bits must be positive: $exp" >&2
    exit 1
  fi
done

for man in "${MAN_BITS[@]}"; do
  if ! [[ "$man" =~ ^-?[0-9]+$ ]]; then
    echo "Invalid mantissa bit entry: $man" >&2
    exit 1
  fi
  if (( man < 0 )); then
    echo "Mantissa bits must be non-negative: $man" >&2
    exit 1
  fi
done

echo "Sweeping uniform weight bit-widths: ${BITS[*]}"
echo "Sweeping floating-point configurations for exponent bits: ${EXP_BITS[*]}, mantissa bits: ${MAN_BITS[*]}"
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

mkdir -p "$UNIFORM_SWEEP_ROOT"
mkdir -p "$FLOAT_SWEEP_ROOT"

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

step=4
for bit in "${BITS[@]}"; do
  QUANT_OUT_DIR="${UNIFORM_SWEEP_ROOT}/${bit}bit"
  mkdir -p "$QUANT_OUT_DIR"

  echo "=== Step ${step}: Quantize to ${bit}-bit weights ==="
  if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
    python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
      --out_dir "$QUANT_OUT_DIR" \
      --num_bits "$bit"
  else
    echo "Found existing ${bit}-bit checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
  fi

  step=$((step + 1))

  echo "=== Step ${step}: Evaluate the ${bit}-bit checkpoint ==="
  python3 sample.py \
    --out_dir "$QUANT_OUT_DIR" \
    --eval_only \
    --eval_dataset minipile

  step=$((step + 1))
done
FLOAT_SPECS=()
for exp in "${EXP_BITS[@]}"; do
  for man in "${MAN_BITS[@]}"; do
    label="e${exp}_m${man}"
    FLOAT_SPECS+=("${label}:${exp}:${man}")
    QUANT_OUT_DIR="${FLOAT_SWEEP_ROOT}/${label}"
    mkdir -p "$QUANT_OUT_DIR"
    total_bits=$((1 + exp + man))

    echo "=== Step ${step}: Quantize to float configuration exp=${exp}, man=${man} (total ${total_bits}-bit) ==="
    if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
      python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
        --out_dir "$QUANT_OUT_DIR" \
        --quantization float \
        --float-exp-bits "$exp" \
        --float-man-bits "$man"
    else
      echo "Found existing float checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
    fi

    step=$((step + 1))

    echo "=== Step ${step}: Evaluate the float checkpoint (exp=${exp}, man=${man}) ==="
    python3 sample.py \
      --out_dir "$QUANT_OUT_DIR" \
      --eval_only \
      --eval_dataset minipile

    step=$((step + 1))
  done
done
python3 - "$OUT_DIR" "$UNIFORM_SWEEP_ROOT" "$FLOAT_SWEEP_ROOT" "${BITS[@]}" -- "${FLOAT_SPECS[@]}" <<'PY'
import json
import os
import sys


def load_eval_loss(path: str) -> float:
    if not os.path.exists(path):
        raise SystemExit(f"Missing evaluation summary at {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    val = data.get("val")
    if val is None:
        raise SystemExit(f"No 'val' key found in {path}")
    return float(val)


def parse_float_specs(specs):
    parsed = []
    for spec in specs:
        try:
            label, exp_str, man_str = spec.split(":")
        except ValueError as exc:  # pragma: no cover - CLI parsing guard
            raise SystemExit(f"Invalid floating-point spec '{spec}': {exc}")
        parsed.append((label, int(exp_str), int(man_str)))
    return parsed


def write_uniform_csv(path: str, baseline_loss: float, entries):
    if not entries:
        return
    with open(path, "w", encoding="utf-8") as csv_file:
        csv_file.write("bits,label,val_loss\n")
        csv_file.write(f"32,fp32,{baseline_loss:.8f}\n")
        for bit, loss, label in entries:
            csv_file.write(f"{bit},{label},{loss:.8f}\n")
    print(f"Wrote uniform summary CSV to {path}")


def write_float_csv(path: str, entries):
    if not entries:
        return
    with open(path, "w", encoding="utf-8") as csv_file:
        csv_file.write("exp_bits,man_bits,total_bits,label,val_loss\n")
        for exp, man, total_bits, loss, label in entries:
            csv_file.write(f"{exp},{man},{total_bits},{label},{loss:.8f}\n")
    print(f"Wrote floating-point summary CSV to {path}")


def write_comparison_csv(path: str, baseline_loss: float, uniform_entries, float_entries):
    with open(path, "w", encoding="utf-8") as csv_file:
        csv_file.write("scheme,descriptor,total_bits,val_loss\n")
        csv_file.write(f"baseline,fp32,32,{baseline_loss:.8f}\n")
        for bit, loss, label in uniform_entries:
            csv_file.write(f"uniform,{label},{bit},{loss:.8f}\n")
        for exp, man, total_bits, loss, _ in float_entries:
            descriptor = f"float e={exp}, m={man}"
            csv_file.write(f"float,{descriptor},{total_bits},{loss:.8f}\n")
    print(f"Wrote combined summary CSV to {path}")


out_dir = os.path.abspath(sys.argv[1])
uniform_root = os.path.abspath(sys.argv[2])
float_root = os.path.abspath(sys.argv[3])
args = sys.argv[4:]

if "--" in args:
    idx = args.index("--")
    uniform_bits = [int(arg) for arg in args[:idx]]
    float_specs = args[idx + 1 :]
else:
    uniform_bits = [int(arg) for arg in args]
    float_specs = []

baseline_path = os.path.join(out_dir, "eval_loss.txt")
baseline_loss = load_eval_loss(baseline_path)

uniform_entries = []
for bit in uniform_bits:
    label = f"{bit}-bit"
    path = os.path.join(uniform_root, f"{bit}bit", "eval_loss.txt")
    loss = load_eval_loss(path)
    uniform_entries.append((bit, loss, label))

float_entries = []
for label, exp, man in parse_float_specs(float_specs):
    total_bits = 1 + exp + man
    path = os.path.join(float_root, label, "eval_loss.txt")
    loss = load_eval_loss(path)
    float_entries.append((exp, man, total_bits, loss, label))

uniform_entries.sort(key=lambda item: item[0], reverse=True)
float_entries.sort(key=lambda item: (item[2], item[0]), reverse=True)

write_uniform_csv(
    os.path.join(uniform_root, "uniform_quantization_eval.csv"),
    baseline_loss,
    uniform_entries,
)
write_float_csv(
    os.path.join(float_root, "floating_quantization_eval.csv"),
    float_entries,
)
write_comparison_csv(
    os.path.join(float_root, "quantization_eval_comparison.csv"),
    baseline_loss,
    uniform_entries,
    float_entries,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    print("matplotlib is not installed; skipping plot generation.")
else:
    plt.figure(figsize=(8, 4.5))
    if uniform_entries:
        bits = [item[0] for item in uniform_entries]
        losses = [item[1] for item in uniform_entries]
        plt.plot(bits, losses, marker="o", label="Uniform checkpoints")
    if float_entries:
        markers = ["s", "^", "D", "P", "X", "*", "o"]
        by_exp = {}
        for exp, man, total_bits, loss, label in float_entries:
            by_exp.setdefault(exp, []).append((total_bits, loss, man))
        for idx, (exp, entries) in enumerate(sorted(by_exp.items(), reverse=True)):
            entries.sort(key=lambda item: item[0], reverse=True)
            bits = [item[0] for item in entries]
            losses = [item[1] for item in entries]
            marker = markers[idx % len(markers)]
            plt.scatter(bits, losses, marker=marker, label=f"Float exp={exp}")
            for bit, loss, man in entries:
                plt.annotate(
                    f"m={man}",
                    (bit, loss),
                    textcoords="offset points",
                    xytext=(0, -12),
                    ha="center",
                    fontsize=8,
                )
    plt.axhline(baseline_loss, linestyle="--", color="tab:orange", label="fp32 baseline")
    plt.gca().invert_xaxis()
    plt.xlabel("Total bits per weight")
    plt.ylabel("Validation loss")
    plt.title("Fake PTQ on minipile: uniform vs floating-point sweeps")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plot_path = os.path.join(float_root, "quantization_eval_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Wrote comparison plot to {plot_path}")
PY
