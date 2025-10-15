#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp_large.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/internal_hosts.yaml \
    --resume_ckpt /home/xinting/Evo_GPT/optimization_and_search/nsga_search/ckpts/infi_medium/ckpt_gen50.json \
    --pop_size 24 \
    --max_layers 24 \
    --min_layers 2 \
    --offspring 12 \
    --generations 1 \
    --exp_name infi_large_try \
    --conda_env reallmforge \
    --max_iters 100 \
    2>&1 | tee -a "$log"
