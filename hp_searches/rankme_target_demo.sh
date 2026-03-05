#!/bin/bash
# hp_searches/rankme_target_demo.sh

set -euo pipefail

for mode in max min; do
  python3 hyperparam_search.py \
    --orig_settings ./hp_searches/efficiency_targets_demo.yaml \
    --param_names \
      n_layer \
      n_head \
      n_embd \
      mlp_size \
    --increments \
      1 \
      1 \
      16 \
      16 \
    --iterations 1 \
    --num_iterations 1 \
    --random_iterations 1 \
    --efficiency_target params \
    --optimize_target rankme \
    --optimize_mode "${mode}" \
    --results_file "rankme_target_${mode}_demo.yaml"
done
