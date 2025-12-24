#!/bin/bash
# zeus_gpu_energy_demo.sh
# Demo: run GPU energy profiling with Zeus during sampling.
# may need to do the following
# python -m pip uninstall -y pynvml py3nvml nvidia-ml-py3
# python -m pip install -U nvidia-ml-py

set -euo pipefail

PROMPT="Once upon a time"


python3 sample.py \
  --init_from resume \
  --device cuda \
  --num_samples 1 \
  --top_k 1 \
  --max_new_tokens 1 \
  --start "${PROMPT}" \
  --zeus_profile \
  --zeus_profile_target gpu

python3 sample.py \
  --init_from resume \
  --device cuda \
  --num_samples 1 \
  --top_k 1 \
  --max_new_tokens 100 \
  --start "${PROMPT}" \
  --zeus_profile \
  --zeus_profile_target gpu

python3 sample.py \
  --init_from resume \
  --device cuda \
  --num_samples 10 \
  --top_k 1 \
  --max_new_tokens 100 \
  --start "${PROMPT}" \
  --zeus_profile \
  --zeus_profile_target gpu
