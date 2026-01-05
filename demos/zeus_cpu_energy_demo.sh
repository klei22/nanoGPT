#!/bin/bash
# zeus_cpu_energy_demo.sh
# Demo: run CPU energy profiling with Zeus during sampling.

set -euo pipefail

PROMPT="Once upon a time"

python3 sample.py \
  --init_from gpt2 \
  --device cpu \
  --num_samples 1 \
  --max_new_tokens 32 \
  --start "${PROMPT}" \
  --zeus_profile \
  --zeus_profile_target cpu
