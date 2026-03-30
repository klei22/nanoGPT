#!/bin/bash
# head_search.sh hp_searches/rubiks_cube_predictor.sh


python3 optimization_and_search/hyperparam_search.py \
  --orig_settings ./hp_searches/rubiks_cube_predictor.yaml \
  --param_names \
        n_layer \
        n_head \
        n_cproj \
        n_embd \
        mlp_size \
  --increments \
        1 \
        1 \
        1 \
        16 \
        16 \
  --random_iterations 1 \
  --iterations 1 \
  --num_iterations 20000 \
  --max_iters_increase 100 \
  --results_file results.yaml

