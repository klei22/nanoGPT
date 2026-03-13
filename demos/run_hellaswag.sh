#!/usr/bin/env bash
set -euo pipefail

# Fill these in once you have the files.
CKPT_PATH="nsga_exps/infi_med_2_8_val_loss/gen52-20251014_170936_882d-host0/gen52-row0/ckpt.pt"
CONFIG_PATH="nsga_exps/infi_med_2_8_val_loss/gen52-20251014_170936_882d-host0/gen52-row0/full_config.json"
OUT_DIR="out_eval"

python3 benchmarks/run_hellaswag.py \
  --ckpt_path "$CKPT_PATH" \
  --config_path "$CONFIG_PATH" \
  --out_dir "$OUT_DIR" \
  --dtype bfloat16 \
  --device cuda \
  --seed 1048 \
  --print_examples \
  --split validation
