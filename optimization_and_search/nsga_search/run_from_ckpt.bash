#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/host_no_east4.yaml \
    --search_space_config search_space_def/default_search_space.yaml \
    --resume_ckpt /home/xinting/Evo_GPT/optimization_and_search/nsga_search/ckpts/infi_medium/pkl/1021_1822_pop_gen100.pkl \
    --pop_size 24 \
    --max_layers 24 \
    --min_layers 2 \
    --offspring 12 \
    --generations 50 \
    --exp_name infi_medium \
    --conda_env reallmforge \
    --max_iters 10000 \
    2>&1 | tee -a "$log"
