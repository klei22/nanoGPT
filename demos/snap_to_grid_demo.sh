#!/bin/bash
# snap_to_grid_demo.sh
# Demonstration for generating and evaluating snap-to-grid registries
# during training and sampling.

set -euo pipefail

OUT_DIR=${OUT_DIR:-out_snap_to_grid_demo}
SNAP_COMPONENTS=${SNAP_COMPONENTS:-both}
# Allow overriding the layers/sizes through environment variables while
# supplying sensible defaults.
read -r -a SNAP_LAYERS <<< "${SNAP_LAYERS:-1 2}"
read -r -a SNAP_SIZES <<< "${SNAP_SIZES:-16 64}"

SNAP_LAYER_ARGS=()
if ((${#SNAP_LAYERS[@]})); then
  SNAP_LAYER_ARGS=(--snap_to_grid_layers "${SNAP_LAYERS[@]}")
fi

SNAP_SIZE_ARGS=()
if ((${#SNAP_SIZES[@]})); then
  SNAP_SIZE_ARGS=(--snap_to_grid_sizes "${SNAP_SIZES[@]}")
fi

printf 'Preparing shakespeare_char dataset...\n'
pushd data/shakespeare_char >/dev/null
bash get_dataset.sh
popd >/dev/null

printf 'Training demo model with snap-to-grid sweeps...\n'
python3 train.py \
  --device cpu \
  --dataset shakespeare_char \
  --out_dir "${OUT_DIR}" \
  --block_size 128 \
  --batch_size 32 \
  --n_layer 3 \
  --n_head 4 \
  --n_embd 192 \
  --max_iters 200 \
  --eval_interval 100 \
  --eval_iters 100 \
  --learning_rate 3e-4 \
  --dropout 0.1 \
  --enable_snap_to_grid \
  --snap_to_grid_components "${SNAP_COMPONENTS}" \
  "${SNAP_LAYER_ARGS[@]}" \
  "${SNAP_SIZE_ARGS[@]}" \
  --sample_each_eval \
  --max_sample_tokens 64 \
  --tensorboard_log \
  --tensorboard_run_name "${OUT_DIR}_snap_demo"

printf 'Evaluating validation loss for each snap-to-grid size...\n'
python3 sample.py \
  --device cpu \
  --out_dir "${OUT_DIR}" \
  --init_from "${OUT_DIR}" \
  --eval_only \
  --eval_iters 100 \
  --enable_snap_to_grid \
  --snap_to_grid_components "${SNAP_COMPONENTS}" \
  "${SNAP_LAYER_ARGS[@]}" \
  "${SNAP_SIZE_ARGS[@]}"

printf 'Generating one sample for each snap-to-grid setting...\n'
python3 sample.py \
  --device cpu \
  --out_dir "${OUT_DIR}" \
  --init_from "${OUT_DIR}" \
  --num_samples 1 \
  --max_new_tokens 80 \
  --temperature 0.8 \
  --enable_snap_to_grid \
  --snap_to_grid_components "${SNAP_COMPONENTS}" \
  "${SNAP_LAYER_ARGS[@]}" \
  "${SNAP_SIZE_ARGS[@]}" \
  --start "\n"
