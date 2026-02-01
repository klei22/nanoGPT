#!/bin/bash
# multicontext_demo.sh
#
# pushd data/shakespeare_char
# bash get_dataset.sh
# popd
#
# pushd data/shakespeare_char_cvp/
# bash get_dataset.sh
# popd
#
# pushd data/shakespeare_char_in_word_position/
# bash get_dataset.sh
# popd
#
# pushd data/shakespeare_char_part_of_speech/
# bash get_dataset.sh
# popd
#
# pushd data/shakespeare_char_since_newline/
# bash get_dataset.sh
# popd
      # shakespeare_char \
      # shakespeare_char_case_map \
      # shakespeare_char_lowercase \
      # shakespeare_char_cvp \
      # shakespeare_char_in_word_position \
      # shakespeare_char_part_of_speech \
      # shakespeare_char_since_newline \
#
python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets \
      shakespeare_char \
      shakespeare_char_case_map \
      shakespeare_char_lowercase \
      shakespeare_char_cvp \
      shakespeare_char_in_word_position \
      shakespeare_char_part_of_speech \
   --max_iters 2000 \
   --dropout 0.3 \
   --use_rotary_embeddings \
   --no-use_abs_pos_embeddings \
   --out_dir ./fl_out_weight_tying \
   --compile

python3 sample.py \
  --out_dir ./fl_out_weight_tying \
  --multicontext \
  --multicontext_datasets \
    shakespeare_char \
    shakespeare_char_case_map \
    shakespeare_char_lowercase \
    shakespeare_char_cvp \
    shakespeare_char_in_word_position \
    shakespeare_char_part_of_speech \
  --multicontext_start \
    "But" \
    "ULL" \
    "but" \
    "323" \
    "124" \
    "fff" \
  --max_new_tokens 256 \
  --top_k 1 \
  --num_samples 1

    # "Hello " \
    # "ULLLL_" \
    # "hello " \
    # "21221_" \
    # "12345_" \
    # "aaaaa " \
    # "123456" \
