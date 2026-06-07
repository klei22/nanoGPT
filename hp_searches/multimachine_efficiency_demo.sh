#!/usr/bin/env bash
# hp_searches/multimachine_efficiency_demo.sh
# Multi-machine hp_search demo. Run from the repository root after setting:
#   HP_SEARCH_HOSTS="10.0.0.11 10.0.0.12" bash hp_searches/multimachine_efficiency_demo.sh
# Optional shared defaults:
#   HP_SEARCH_USER=ubuntu
#   HP_SEARCH_REMOTE_WORK_DIR=/home/ubuntu/Evo_GPT
#   HP_SEARCH_CONDA_ENV=reallmforge
# Optional per-host overrides (same order/count as HP_SEARCH_HOSTS):
#   HP_SEARCH_USERS="ubuntu nvidia"
#   HP_SEARCH_REMOTE_WORK_DIRS="/home/ubuntu/Evo_GPT /home/nvidia/nanoGPT"
#   HP_SEARCH_CONDA_ENVS="reallmforge nanojetson"
# Other optional overrides:
#   HP_SEARCH_RESULTS_FILE=multimachine_efficiency_results.yaml
#   HP_SEARCH_TIMEOUT=86400
#   HP_SEARCH_OVERRIDE_CFG="device=cuda:0 dtype=float16 compile=False batch_size=16"

set -euo pipefail

if [[ -z "${HP_SEARCH_HOSTS:-}" ]]; then
  cat >&2 <<'EOF'
HP_SEARCH_HOSTS is required and should be a whitespace-separated list, for example:
  HP_SEARCH_HOSTS="10.0.0.11 10.0.0.12" bash hp_searches/multimachine_efficiency_demo.sh
EOF
  exit 2
fi

read -r -a HP_SEARCH_HOST_ARRAY <<< "${HP_SEARCH_HOSTS}"
HP_SEARCH_EFFECTIVE_USER="${HP_SEARCH_USER:-${USER}}"
HP_SEARCH_EFFECTIVE_REMOTE_WORK_DIR="${HP_SEARCH_REMOTE_WORK_DIR:-/home/${HP_SEARCH_EFFECTIVE_USER}/Evo_GPT}"
read -r -a HP_SEARCH_OVERRIDE_CFG_ARRAY <<< "${HP_SEARCH_OVERRIDE_CFG:-}"

cmd=(
  python3 hyperparam_search.py
  --orig_settings "${HP_SEARCH_ORIG_SETTINGS:-./hp_searches/multimachine_efficiency_demo.yaml}"
  --param_names
    n_layer
    n_head
    n_kv_group
    n_embd
    mlp_size
    n_qk_head_dim
    n_v_head_dim
  --increments
    1
    1
    1
    32
    32
    32
    32
  --random_iterations "${HP_SEARCH_RANDOM_ITERATIONS:-2}"
  --iterations "${HP_SEARCH_ITERATIONS:-1}"
  --num_iterations "${HP_SEARCH_NUM_ITERATIONS:-25}"
  --efficiency_target "${HP_SEARCH_EFFICIENCY_TARGET:-params}"
  --max_iters_increase "${HP_SEARCH_MAX_ITERS_INCREASE:-1000}"
  --results_file "${HP_SEARCH_RESULTS_FILE:-multimachine_efficiency_results.yaml}"
  --override_cfg "${HP_SEARCH_OVERRIDE_CFG_ARRAY[@]}"
  --distributed_hosts "${HP_SEARCH_HOST_ARRAY[@]}"
  --distributed_user "${HP_SEARCH_EFFECTIVE_USER}"
  --distributed_conda_env "${HP_SEARCH_CONDA_ENV:-reallmforge}"
  --distributed_run_dir_name "${HP_SEARCH_RUN_DIR_NAME:-hp_search_multimachine_efficiency}"
  --distributed_poll_interval "${HP_SEARCH_POLL_INTERVAL:-60}"
  --distributed_timeout "${HP_SEARCH_TIMEOUT:-86400}"
)

if [[ -n "${HP_SEARCH_USERS:-}" ]]; then
  read -r -a HP_SEARCH_USER_ARRAY <<< "${HP_SEARCH_USERS}"
  cmd+=(--distributed_users "${HP_SEARCH_USER_ARRAY[@]}")
fi

if [[ -n "${HP_SEARCH_REMOTE_WORK_DIRS:-}" ]]; then
  read -r -a HP_SEARCH_REMOTE_WORK_DIR_ARRAY <<< "${HP_SEARCH_REMOTE_WORK_DIRS}"
  cmd+=(--distributed_remote_work_dirs "${HP_SEARCH_REMOTE_WORK_DIR_ARRAY[@]}")
elif [[ -n "${HP_SEARCH_REMOTE_WORK_DIR:-}" || -z "${HP_SEARCH_USERS:-}" ]]; then
  cmd+=(--distributed_remote_work_dir "${HP_SEARCH_EFFECTIVE_REMOTE_WORK_DIR}")
fi

if [[ -n "${HP_SEARCH_CONDA_ENVS:-}" ]]; then
  read -r -a HP_SEARCH_CONDA_ENV_ARRAY <<< "${HP_SEARCH_CONDA_ENVS}"
  cmd+=(--distributed_conda_envs "${HP_SEARCH_CONDA_ENV_ARRAY[@]}")
fi

"${cmd[@]}"
