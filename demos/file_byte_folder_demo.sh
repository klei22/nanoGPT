#!/bin/bash
set -e

# Demonstration of tokenizing a folder of files, training, and generating files
DATA_DIR=data/file_byte_demo
INPUT_DIR=$DATA_DIR/input
OUT_DIR=out_file_byte_demo

rm -rf $DATA_DIR $OUT_DIR
mkdir -p $INPUT_DIR
mkdir -p $DATA_DIR/generated

# Create sample files with simple patterns
for i in {0..19}; do
  printf "file_%d\n" $i > $INPUT_DIR/file_${i}.txt
done

# Tokenize the folder using the file_byte tokenizer
python data/template/prepare.py --method file_byte -t $INPUT_DIR --train_output $DATA_DIR/train.bin --val_output $DATA_DIR/val.bin --percentage_train 0.9

# Train a tiny model on the dataset
python train.py --device=cpu --out_dir $OUT_DIR --dataset file_byte_demo \
  --batch_size 4 --block_size 16 --n_layer 2 --n_head 2 --n_embd 64 \
  --max_iters 50 --lr_decay_iters 50 --eval_interval 10 --eval_iters 10 --dropout 0.0

# Sample from the model and recover generated files
python sample.py --out_dir $OUT_DIR --num_samples 5 --max_new_tokens 200 --start "" \
  --file_output_dir $DATA_DIR/generated --temperature 1.0 --top_k 200

echo "Generated files:"
ls $DATA_DIR/generated

