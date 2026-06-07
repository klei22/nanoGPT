#!/bin/bash
# demos/linux_commands_colorized_tui_demo.sh
#
# End-to-end demo:
# 1) Build a small segment of the linux-commands dataset
# 2) Train briefly with validation-passage colorization enabled
# 3) Launch the passage timeline TUI

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="shakespeare_char"
OUT_DIR="out/linux_commands_colorize_demo"
YAML_FILE="${OUT_DIR}/highlighted_passage.yaml"

pushd "data/${DATASET}" >/dev/null

# Fetch english split if needed (creates input.txt)
if [[ ! -f input.txt ]]; then
  bash get_dataset.sh eng
fi

# Use only a small segment so the demo runs quickly
head -n 1200 input.txt > demo_segment.txt

# Tokenize the segment into train.bin/val.bin for this dataset
# python3 prepare.py \
#   -t demo_segment.txt \
#   --method tiktoken \
#   --train_output train.bin \
#   --val_output val.bin \
#   -p 0.9
# python3 prepare.py \
#   -t demo_segment.txt \
#   --method tiktoken \
#   --train_output train.bin \
#   --val_output val.bin \
#   -p 0.9

popd >/dev/null

python3 train.py \
  --dataset "${DATASET}" \
  --out_dir "${OUT_DIR}" \
  --device "cuda:0" \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --block_size 256 \
  --batch_size 64 \
  --max_iters 2000 \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --norm_variant_wte rmsnorm \
  --use_qk_norm \
  --use_qk_norm_scale \
  --eval_interval 50 \
  --eval_iters 100 \
  --log_interval 20 \
  --compile \
  --no-tensorboard_log \
  --colorize_val_passage \
  --colorize_val_tokens 256 \
  --colorize_val_offset 0 \
  --colorize_val_mode softmax \
  --colorize_val_yaml highlighted_passage.yaml

echo "Launching TUI viewer for ${YAML_FILE}"
python3 view_colorized_passages.py --yaml_file "${YAML_FILE}"
