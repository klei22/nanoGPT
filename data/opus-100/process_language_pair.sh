#!/bin/bash

set -e

# Only do once
training_text="total_text.txt"
python3 interleave_files.py -f1 en_pho.txt -f2 es_pho.txt -m 50 -o test --forbidden_strings "(en)" "(es)"
cat ./interleaved_files/*.txt > "${training_text}"

# Loop over different sentencepiece tokenizations
for (( i = 512; i < 2048; i+=256 )); do
  target_dir="${i}_bin"
  # Train sentence piece on total 
  python3 prepare.py -t "${training_text}" --method sentencepiece --vocab_size "${i}"
  python3 batch_prepare.py --input_dir interleaved_files --prepare_script prepare.py --tokenizer sentencepiece
  mkdir "$target_dir"

  # move training files to folder
  mv train.bin "$target_dir"
  mv val.bin "$target_dir"
  mv trained_spm_model.model "$target_dir"
  mv trained_spm_model.vocab "$target_dir"
  mv meta.pkl "$target_dir"

  # remove old bin files
  rm ./interleaved_files/*.bin
done
