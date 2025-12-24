#!/bin/bash
# zeus_gpu_energy_demo.sh
# Demo: run GPU energy profiling with Zeus during sampling.

set -euo pipefail

PROMPT="Once upon a time"

python3 sample.py \
  --init_from gpt2 \
  --device cuda \
  --num_samples 1 \
  --max_new_tokens 64 \
  --start "${PROMPT}" \
  --zeus_profile \
  --zeus_profile_target gpu
