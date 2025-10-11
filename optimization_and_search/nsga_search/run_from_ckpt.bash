#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/hosts_used.yaml \
    --pop_size 16 \
    --max_layers 16 \
    --min_layers 1 \
    --offspring 8 \
    --generations 15 \
    --exp_name infi_attn_exp_sample \
    --conda_env reallmforge \
    --max_iters 10000 \
    2>&1 | tee -a "$log"
