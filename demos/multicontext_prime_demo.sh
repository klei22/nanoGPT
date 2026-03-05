#!/bin/bash
# multicontext_prime_demo.sh

pushd data/shakespeare_char_prime
bash get_dataset.sh
popd

 python3 train.py \
   --training_mode multicontext \
   --multicontext \
   --multicontext_datasets \
       shakespeare_char \
       data/shakespeare_char_prime/char_mod2 \
       data/shakespeare_char_prime/char_mod3 \
       data/shakespeare_char_prime/char_mod7 \
       data/shakespeare_char_prime/char_mod11 \
       data/shakespeare_char_prime/char_mod13 \
       data/shakespeare_char_prime/char_mod17 \
    --max_iters 2000 \
    --dropout 0.2 \
    --top_k 1 \
    --sample_each_eval \
    --use_qk_norm \
    --use_qk_norm_scale \
    --no-use_abs_pos_embeddings \
    --use_rotary_embeddings \
    --out_dir ./out_mc_shakespeare \
    --compile

python3 sample.py \
  --out_dir ./out_mc_shakespeare \
  --multicontext \
  --multicontext_datasets \
       shakespeare_char \
       shakespeare_char_prime/char_mod2 \
       shakespeare_char_prime/char_mod3 \
       shakespeare_char_prime/char_mod7 \
       shakespeare_char_prime/char_mod11 \
       shakespeare_char_prime/char_mod13 \
       shakespeare_char_prime/char_mod17 \
  --multicontext_start \
    "But" \
    "aba" \
    "abc" \
    "abc" \
    "abc" \
    "abc" \
    "abc" \
  --max_new_tokens 512 \
  --top_k 1 \
  --num_samples 3

