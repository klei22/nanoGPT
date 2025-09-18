#!/bin/bash
# demos/fake_ptq_interactive_demo.sh
#
# Demonstrates the per-tensor configuration and interactive TUI flow for the
# fake PTQ utility. The script trains a compact Shakespeare character model,
# seeds a few per-tensor bit overrides, then drives the interactive selector to
# refine the configuration before quantizing the checkpoint.

set -euo pipefail

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd data/shakespeare_char > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

OUT_DIR="out_fake_ptq_shakespeare"
QUANTIZED_OUT_DIR="${OUT_DIR}_ptq"

echo "=== Step 2: Train a reference model on shakespeare_char ==="
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "$OUT_DIR" \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --block_size 128 \
  --batch_size 64 \
  --max_iters 500 \
  --lr_decay_iters 500 \
  --eval_iters 50 \
  --log_interval 10 \
  --always_save_checkpoint

echo "=== Step 3: Evaluate the baseline checkpoint ==="
python3 sample.py \
  --device cpu \
  --out_dir "$OUT_DIR" \
  --eval_only \
  --eval_dataset shakespeare_char \
  --eval_iters 200

PER_TENSOR_FILE="$OUT_DIR/per_tensor_bits.json"
TENSOR_LIST="$OUT_DIR/per_tensor_tensor_names.txt"

echo "=== Step 4: Create per-tensor override seeds ==="
python3 - <<PY
import json
import os
import torch

out_dir = "${OUT_DIR}"
ckpt_path = os.path.join(out_dir, "ckpt.pt")
if not os.path.exists(ckpt_path):
    raise SystemExit(f"Checkpoint not found at {ckpt_path}.")

checkpoint = torch.load(ckpt_path, map_location="cpu")
if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

if not isinstance(state_dict, dict):
    raise SystemExit("Unsupported checkpoint format: expected a state dict.")

overrides = {}
selected = []
for name, tensor in state_dict.items():
    if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
        continue
    if len(selected) == 0:
        overrides[name] = 6
    elif len(selected) == 1:
        overrides[name] = 4
    else:
        overrides[name] = 0
    selected.append(name)
    if len(selected) == 3:
        break

if not overrides:
    raise SystemExit("No floating-point tensors found to override.")

os.makedirs(out_dir, exist_ok=True)
with open("${PER_TENSOR_FILE}", "w", encoding="utf-8") as handle:
    json.dump(overrides, handle, indent=2)
with open("${TENSOR_LIST}", "w", encoding="utf-8") as handle:
    for name in selected:
        handle.write(name + "\n")

print("Seeded overrides for:")
for name in selected:
    print(f"  - {name}")
PY

mapfile -t SELECTED_TENSORS < "$TENSOR_LIST"
FIRST_TENSOR="${SELECTED_TENSORS[0]}"
SECOND_TENSOR="${SELECTED_TENSORS[1]:-}"
THIRD_TENSOR="${SELECTED_TENSORS[2]:-}"

TUI_COMMANDS="$OUT_DIR/tui_commands.txt"
{
  echo "all 6"
  if [ -n "$FIRST_TENSOR" ]; then
    echo "set $FIRST_TENSOR 5"
  fi
  if [ -n "$SECOND_TENSOR" ]; then
    echo "set $SECOND_TENSOR 3"
  fi
  if [ -n "$THIRD_TENSOR" ]; then
    echo "set $THIRD_TENSOR 0"
  fi
  echo "apply"
} > "$TUI_COMMANDS"

echo "Interactive command script:"
cat "$TUI_COMMANDS"

echo "=== Step 5: Apply fake PTQ with interactive refinement ==="
python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
  --out_dir "$QUANTIZED_OUT_DIR" \
  --num_bits 8 \
  --per-tensor-bits "$PER_TENSOR_FILE" \
  --quantization asymmetric \
  --interactive \
  --min-bits 2 \
  --max-bits 8 \
  --tui-page-size 12 < "$TUI_COMMANDS"

echo "=== Step 6: Evaluate the quantized checkpoint ==="
python3 sample.py \
  --device cpu \
  --out_dir "$QUANTIZED_OUT_DIR" \
  --eval_only \
  --eval_dataset shakespeare_char \
  --eval_iters 200
