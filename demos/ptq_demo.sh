#!/bin/bash
# demos/ptq_demo.sh

# 1. Prepare minipile dataset
pushd data/minipile
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for minipile"
fi
popd

# 2. Train a larger model on minipile
out_dir="out_ptq_demo"
run_name_before="ptq_fp32"
python3 train.py \
  --dataset minipile \
  --out_dir "$out_dir" \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --block_size 256 \
  --max_iters 10000 \
  --log_interval 10 \
  --tensorboard_run_name "$run_name_before"

# 3. Compute model stats before quantization
python3 train.py \
  --dataset minipile \
  --out_dir "$out_dir" \
  --eval_only \
  --compute_model_stats \
  --print_model_stats_table "${run_name_before}.csv" \
  --tensorboard_run_name "$run_name_before"

# 4. Report validation loss before quantization
python3 sample.py \
  --out_dir "$out_dir" \
  --eval_only \
  --eval_dataset minipile

# 5. Sample from the original model
python3 sample.py \
  --out_dir "$out_dir" \
  --num_samples 1 \
  --max_new_tokens 50 \
  --start "Hello" \
  --sample_file before_ptq.txt

# 6. Apply fake PTQ (8-bit uniform)
python3 quantizations/ptq/fake_quantize_ckpt.py "$out_dir" --num_bits 8 --out_dir "${out_dir}_ptq"

# 7. Compute model stats after quantization
run_name_after="ptq_int8"
python3 train.py \
  --dataset minipile \
  --out_dir "${out_dir}_ptq" \
  --eval_only \
  --compute_model_stats \
  --print_model_stats_table "${run_name_after}.csv" \
  --tensorboard_run_name "$run_name_after"

# 8. Report validation loss after quantization
python3 sample.py \
  --out_dir "${out_dir}_ptq" \
  --eval_only \
  --eval_dataset minipile

# 9. Sample from the quantized model
python3 sample.py \
  --out_dir "${out_dir}_ptq" \
  --num_samples 1 \
  --max_new_tokens 50 \
  --start "Hello" \
  --sample_file after_ptq.txt

# 10. Evaluate tolerance to uniform angular noise on validation loss
ANGULAR_NOISE_SWEEP="${ANGULAR_NOISE_SWEEP:-5 15 30}"
if [ -n "${ANGULAR_NOISE_SWEEP// }" ]; then
  read -r -a ANGULAR_NOISE_LEVELS <<< "$ANGULAR_NOISE_SWEEP"
else
  ANGULAR_NOISE_LEVELS=()
fi

if [ "${#ANGULAR_NOISE_LEVELS[@]}" -gt 0 ]; then
  echo "=== Evaluating uniform angular noise tolerance (${ANGULAR_NOISE_LEVELS[*]} degrees) ==="
  declare -a ANGULAR_NOISE_DIRS=()
  for angle in "${ANGULAR_NOISE_LEVELS[@]}"; do
    sanitized_angle=${angle//./p}
    sanitized_angle=${sanitized_angle//-/_}
    noise_dir="${out_dir}_ptq_noise_${sanitized_angle}deg"
    echo "--- Applying ${angle}-degree angular noise and evaluating validation loss ---"
    rm -rf "$noise_dir"
    cp -r "${out_dir}_ptq" "$noise_dir"
    python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
      "${noise_dir}/ckpt.pt" \
      '.*\.weight$' \
      --device cpu \
      --no-colorize \
      --angular-noise-max "$angle" \
      --angular-noise-units degrees \
      --angular-noise-pattern '.*\.weight$' \
      --angular-noise-output "${noise_dir}/ckpt.pt" \
      --angular-noise-seed 42 \
      --angular-noise-only
    python3 sample.py \
      --out_dir "$noise_dir" \
      --eval_only \
      --eval_dataset minipile
    ANGULAR_NOISE_DIRS+=("$noise_dir")
  done

  python3 - "${out_dir}_ptq" "${ANGULAR_NOISE_LEVELS[@]}" -- "${ANGULAR_NOISE_DIRS[@]}" <<'PY'
import json
import os
import sys

args = sys.argv[1:]
if "--" in args:
    sep_index = args.index("--")
    noise_levels = args[1:sep_index]
    noise_dirs = args[sep_index + 1 :]
else:
    noise_levels = args[1:]
    noise_dirs = []

if len(noise_levels) != len(noise_dirs):
    raise SystemExit(
        "Mismatch between angular noise sweep values and evaluation directories"
    )

ptq_dir = os.path.abspath(args[0])
results = []

baseline_path = os.path.join(ptq_dir, "eval_loss.txt")
if os.path.exists(baseline_path):
    with open(baseline_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if "val" in data:
        results.append((0.0, float(data["val"]), os.path.basename(ptq_dir)))

for level, directory in zip(noise_levels, noise_dirs):
    eval_dir = os.path.abspath(directory)
    eval_path = os.path.join(eval_dir, "eval_loss.txt")
    if not os.path.exists(eval_path):
        raise SystemExit(f"Missing evaluation summary at {eval_path}")
    with open(eval_path, encoding="utf-8") as fh:
        data = json.load(fh)
    val = data.get("val")
    if val is None:
        raise SystemExit(f"No 'val' key found in {eval_path}")
    results.append((float(level), float(val), os.path.basename(eval_dir)))

if not results:
    raise SystemExit("No angular noise evaluation results collected")

results.sort(key=lambda item: item[0])
csv_path = os.path.join(ptq_dir, "angular_noise_eval.csv")
with open(csv_path, "w", encoding="utf-8") as csv_file:
    csv_file.write("angle_degrees,label,val_loss\n")
    for angle, loss, label in results:
        csv_file.write(f"{angle},{label},{loss:.8f}\n")

print(f"Wrote angular noise evaluation summary to {csv_path}")
PY
else
  echo "Skipping angular noise sweep; set ANGULAR_NOISE_SWEEP to space-separated angles to enable."
fi
