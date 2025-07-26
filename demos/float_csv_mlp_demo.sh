#!/bin/bash
# demos/float_csv_mlp_demo.sh
# Demonstrate preparing a float CSV dataset for multicontext training using the index MLP.

set -e

# Generate a simple CSV of sine and cosine waves
python3 - <<'PY'
import numpy as np
x = np.linspace(0, 4*np.pi, 500)
arr = np.stack([np.sin(x), np.cos(x)], axis=1)
np.savetxt('sinewaves.csv', arr, delimiter=',')
PY

# Tokenize the CSV into separate datasets under data/
python3 data/template/prepare.py \
  --method float_csv \
  --csv_file sinewaves.csv \
  --csv_prefix data/sine_demo \
  --csv_percentage_train 0.9

# Train a small model on the generated datasets using the index MLP
python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --mc_use_index_mlp \
  --multicontext_datasets sine_demo_0 sine_demo_1 \
  --block_size 32 \
  --batch_size 16 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --max_iters 200 \
  --compile

# Clean up the temporary csv
rm sinewaves.csv
