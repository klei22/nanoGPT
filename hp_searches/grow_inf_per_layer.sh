#!/bin/bash
# hp_searches/grow_inf_per_layer.sh

python3 optimization_and_search/hyperparam_search.py \
  --orig_settings ./hp_searches/grow_inf_per_layer.yaml \
  --param_names \
        n_layer \
        n_head_layerlist \
        n_embd \
        mlp_size_layerlist \
        n_qk_head_dim_layerlist \
        n_v_head_dim_layerlist \
  --increments \
        1 \
        1 \
        16 \
        16 \
        16 \
        16 \
  --iterations 1 \
  --random_iterations 1 \
  --num_iterations 10000 \
  --max_iters_increase 1000 \
  --nlayer_dup_mode dup_each \
  --results_file sweep_log.yaml

