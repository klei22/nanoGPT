#!/bin/bash

for (( i = 1000; i <= 10000; i+=1000 )); do

  python3 prepare.py \
    -t factor_train.txt \
    -v factor_val.txt \
    --method char_bpe \
    --vocab_size "${i}" \
    -T -s -S "factored_${i}"

  mv char_bpe_vocab.json "char_bpe_factored_${i}/"
  mv char_bpe_token_counts.json "char_bpe_factored_${i}/"
done

for (( i = 1000; i <= 10000; i+=1000 )); do
  python3 prepare.py \
    -t input_train.txt \
    -v input_val.txt \
    --method char_bpe \
    --vocab_size "${i}" \
    -T -s -S "unfactored_${i}"

  mv char_bpe_vocab.json "char_bpe_unfactored_${i}/"
  mv char_bpe_token_counts.json "char_bpe_unfactored_${i}/"

done

