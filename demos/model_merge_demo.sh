#!/usr/bin/env bash
set -euo pipefail

# Demo commands for model_merge.py. Replace paths with real checkpoint dirs.
CKPT_A="out/run_a"
CKPT_B="out/run_b"

echo "==> L2-normalized merge (default)"
python3 model_merge.py "${CKPT_A}" "${CKPT_B}" --out_dir out/merge_l2

echo "==> L2-normalized merge but skip final norm for wte/lm_head"
python3 model_merge.py "${CKPT_A}" "${CKPT_B}" \
  --out_dir out/merge_skip_final_norm \
  --skip_final_norm_wte_lm_head

echo "==> Simple averaging without any L2 normalization"
python3 model_merge.py "${CKPT_A}" "${CKPT_B}" \
  --out_dir out/merge_simple_avg \
  --no_l2_normalize \
  --simple_divisor 2.0
