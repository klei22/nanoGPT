#/bin/bash

# head to repo root
cd ../

# head to csv_data folder
pushd data/csv_data
# create data
bash main.sh

# create train.bin and val.bin splits (retaining contiguous sections of data)
python3 prepare.py -i processed_sine_data.csv

# return to repo root
popd

# start training
python3 train.py \
  --max_iters 3000 \
  --eval_interval 300 \
  --eval_iters 200 \
  --log_interval 10 \
  --dataset csv_data \
  --tensorboard_project csv_data \
  --tensorboard_run_name csv_data \
  --block_size 1000  \
  --out_dir csv_explorations \
  --compile

# start inference
# TODO: seed with context from validation set, or new data set
python3 sample.py \
  --device "cuda" \
  --num_samples  1 \
  --max_new_tokens 1000 \
  --out_dir csv_explorations \
  --temperature 1.0 | tee forecast.txt

# TODO: plot forecast.txt against validation

