#!/bin/bash
# hp_searches/distributed_shakespeare_char.sh
# Example distributed hp_search invocation. Set HP_SEARCH_HOSTS to a whitespace
# separated host list, e.g.:
#   HP_SEARCH_HOSTS="10.0.0.11 10.0.0.12" bash hp_searches/distributed_shakespeare_char.sh

read -r -a HP_SEARCH_HOST_ARRAY <<< "${HP_SEARCH_HOSTS:-}"

python3 hyperparam_search.py \
  --orig_settings ./hp_searches/shakespeare_char.yaml \
  --param_names n_layer n_head n_embd mlp_size \
  --increments 1 1 32 32 \
  --iterations 1 \
  --random_iterations 2 \
  --num_iterations 10 \
  --efficiency_target params \
  --results_file distributed_results.yaml \
  --distributed_hosts "${HP_SEARCH_HOST_ARRAY[@]}" \
  --distributed_user "${HP_SEARCH_USER:-$USER}" \
  --distributed_remote_work_dir "${HP_SEARCH_REMOTE_WORK_DIR:-/home/${HP_SEARCH_USER:-$USER}/Evo_GPT}" \
  --distributed_conda_env "${HP_SEARCH_CONDA_ENV:-reallmforge}" \
  --distributed_run_dir_name "${HP_SEARCH_RUN_DIR_NAME:-hp_search_shakespeare}"
