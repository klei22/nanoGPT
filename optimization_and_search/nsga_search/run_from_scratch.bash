#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp_large.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/host_config_30s.yaml \
    --max_layers 32 \
    --min_layers 2 \
    --pop_size 20 \
    --offspring 10 \
    --generations 2 \
    --exp_name infi_large_with_qkv_norm_try \
    --conda_env reallmforge \
    --max_iters 100 \
    2>&1 | tee -a "$log"