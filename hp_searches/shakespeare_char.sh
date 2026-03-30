#!/bin/bash
# hp_searches/shakespeare_char.sh

python3 optimization_and_search/hyperparam_search.py \
  --orig_settings ./hp_searches/shakespeare_char.yaml \
  --param_names \
        n_layer \
        n_head \
        n_embd \
        mlp_size_layerlist \
  --increments \
        1 \
        1 \
        16 \
        16 \
  --iterations 1 \
  --random_iterations 1 \
  --num_iterations 100 \
  --nlayer_dup_mode dup_each \
  --results_file sweep_log.yaml

