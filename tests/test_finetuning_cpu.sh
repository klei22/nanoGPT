#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
bash "data/${dataset}/get_dataset.sh"

n_layer="2"
n_head="2"
n_kv_group="2"
n_embd="60"
max_iters="50"
block_size="32"
eval_iters="50"
eval_interval="50"
timestamp="$(date +%F_%T)"
notes="test_finetuning_using_init-from_prev-ckpt"
run_name="${dataset}_finetuning_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"
output_dir="results/${timestamp}_${notes}_finetuning"
output_dir2="results/${timestamp}_${notes}_finetuning_relu"
if [ ! -d "${output_dir}" ]; then
  mkdir -p "${output_dir}"
fi

python3 train.py \
  --max_iters "$max_iters" \
  --n_layer "$n_layer" \
  --n_head "$n_head" \
  --n_kv_group "$n_kv_group" \
  --n_embd "$n_embd" \
  --eval_iters "$eval_iters" \
  --eval_interval "$eval_interval" \
  --log_interval 10 \
  --device cpu \
  --dataset "$dataset" \
  --softmax_variant_attn strongermax \
  --tensorboard_run_name "$run_name" \
  --block_size "$block_size" \
  --bias \
  --out_dir "${output_dir}"

python3 train.py \
  --max_iters "$max_iters" \
  --init_from prev_run \
  --prev_run_ckpt "${output_dir}" \
  --n_layer "$n_layer" \
  --n_head "$n_head" \
  --n_kv_group "$n_kv_group" \
  --n_embd "$n_embd" \
  --eval_iters "$eval_iters" \
  --eval_interval "$eval_interval" \
  --log_interval 10 \
  --device cpu \
  --dataset "$dataset" \
  --softmax_variant_attn relumax \
  --tensorboard_run_name "$run_name" \
  --block_size "$block_size" \
  --bias \
  --out_dir "${output_dir2}"

sleep 3
