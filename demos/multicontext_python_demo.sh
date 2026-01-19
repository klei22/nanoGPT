#!/bin/bash
# multicontext_python_demo.sh

python3 train.py \
  --dataset data/mc_out/ \
  --max_iters 2000 \
  --eval_interval 2000 \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_peri_ln \
  --activation_variant squared_relu \
  --compile \
  --out_dir ./out_orig \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings

python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets \
      data/mc_out/ \
      data/mc_ga/ \
      data/mc_pna/ \
   --max_iters 2000 \
   --eval_interval 2000 \
   --use_qk_norm \
   --use_qk_norm_scale \
   --use_peri_ln \
   --activation_variant squared_relu \
   --compile \
   --out_dir ./out_multicontext \
   --use_rotary_embeddings \
   --no-use_abs_pos_embeddings


for temp in 0.25 0.5 0.75 1.0; do
python3 sample.py \
  --out_dir ./out_orig \
  --start \
    $'"""\nwrite a script that calculates the fibonacci sequence\n"""' \
  --max_new_tokens 256 \
  --top_k 10 \
  --temperature "$temp"

python3 sample.py \
  --out_dir ./out_multicontext \
  --multicontext \
  --multicontext_datasets \
      mc_out \
      mc_ga \
      mc_pna \
  --multicontext_start \
    $'"""\nwrite a script that calculates the fibonacci sequence\n"""' \
    $'BBB\nBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\nBBB' \
    $'   \n                                                     \n   ' \
  --max_new_tokens 256 \
  --top_k 10 \
  --temperature "$temp"
done
