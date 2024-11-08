#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

n_layer="2"
n_head="2"
n_kv_group="2"
n_embd="60"
max_iters="50"
block_size="32"
eval_iters="50"
quant_method="symmetric_quant"
linear_variant="quantized_linear"
eval_interval="50"
timestamp="$(date +%F_%T)"
notes="check_all_quantization_options"
run_name="${dataset}_quantization_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"

output_dir="results/${timestamp}_${notes}_quantization"
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
  --quantize_linear_method "$quant_method" \
  --activations_quant_method "$quant_method" \
  --quantization_warmup_iters 0 \
  --quantize_attn_act \
  --quantize_mlp_act \
  --linear_variant_attn "$linear_variant" \
  --linear_variant_mlp "$linear_variant" \
  --store_activations \
  --tensorboard_run_name "$run_name" \
  --block_size "$block_size" \
  --out_dir "${output_dir}"

python3 sample.py \
  --out_dir "${output_dir}" \
  --device "cpu" \
  --num_samples 1 \
  --max_new_tokens 100 \
  --start "What great fortune this is"

python3 quantization/save_weights.py \
  --out_dir "${output_dir}" \
  --file_name "quantized_data" \
  --file_type "pkl" \
  --device "cpu"

python3 quantization/visualize.py \
--file_name "quantized_data.pkl" \
--image_folder "quantized_images" \
--weight "all" \
--graph="histogram"

sleep 3
