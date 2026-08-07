#!/bin/bash

python3 hanzi_bpe_pretokenizer.py \
  encode \
  --ids IDS_BabelStone16.txt \
  --region T \
  --families all \
  --report encode.json \
  input_train.txt factor_train.txt

python3 hanzi_bpe_pretokenizer.py \
  encode \
  --ids IDS_BabelStone16.txt \
  --region T \
  --families all \
  --report encode.json \
  input_val.txt factor_val.txt

# python3 hanzi_bpe_pretokenizer.py \
#   decode \
#   --ids IDS_BabelStone16.txt \
#   --region T \
#   --families all \
#   --report decode.json \
#   corpus.factor.txt corpus.roundtrip.txt
