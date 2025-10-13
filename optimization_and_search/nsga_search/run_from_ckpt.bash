#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/internal_hosts.yaml \
    --resume_ckpt /home/xinting/Evo_GPT/optimization_and_search/nsga_search/ckpts/infi_attn_exp_2_continued/10090627_pop_gen80.pkl \
    --pop_size 32 \
    --max_layers 24 \
    --min_layers 2 \
    --offspring 16 \
    --generations 50 \
    --exp_name infi_med_try \
    --conda_env reallmforge \
    --max_iters 100 \
    2>&1 | tee -a "$log"
