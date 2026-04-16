#!/usr/bin/env bash
set -euo pipefail

python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --route_mode latin_punct_only \
  --latin_trim_sweep \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --split "validation[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --sweep_examples 2 \
  --example_max_new_tokens 64 \
  --report_dir latin_trim_reports \
  --byte_fallback
