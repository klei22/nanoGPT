#!/bin/bash
# data/sinewave/get_dataset.sh

numeric_encoding="${SINE_NUMERIC_ENCODING:-uint}"
numeric_bitwidth="${SINE_NUMERIC_BITWIDTH:-16}"

for (( i = 0; i < 16; i++ )); do

period=$((i+15))
python prepare.py \
  --method sinewave \
  --train_input dummy.txt \
  --train_output s"$i"/train.bin \
  --val_output s"$i"/val.bin \
  --percentage_train 0.9 \
  --sine_period "$period" \
  --sine_points_per_period 15 \
  --sine_num_periods 2000 \
  --sine_amplitude 50 \
  --sine_numeric_encoding "$numeric_encoding" \
  --sine_numeric_bitwidth "$numeric_bitwidth"

cp meta.pkl ./s"$i"
done
