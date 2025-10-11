#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/internal_hosts.yaml \
    --pop_size 4 \
    --max_layers 16 \
    --min_layers 1 \
    --offspring 2 \
    --generations 1 \
    --exp_name small_trail \
    --conda_env reallmforge \
    --max_iters 100 \
    2>&1 | tee -a "$log"