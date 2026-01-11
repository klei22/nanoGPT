#!/usr/bin/env bash
set -euo pipefail

# Demo: multi-token training mode
#
# This mode runs multiple prediction rounds per batch. Each round predicts the
# next token and then appends the top-1 predicted token back into the input
# sequence (dropping the oldest token) before the next round.
#
# Interpretation:
# - --multi_token_prediction_steps=1 is equivalent to regular next-token
#   prediction (one forward pass, one loss).
# - steps>1 adds additional rounds and combines their losses.
#
# Example (block_size=4, steps=3, using linear loss reduction):
#   input tokens: [t0 t1 t2 t3]
#   targets      : [t1 t2 t3 t4 t5]
#   round 0 predicts [t1..t4] loss * 0
#   round 1 input becomes [t1 t2 t3 top1(t4)] loss * 1
#   round 2 input becomes [t2 t3 top1(t4) top1(t5)] loss * 2
#   total loss = 0*L0 + 1*L1 + 2*L2
#
# Adjust dataset/model flags as needed for your setup.

python train.py \
  --training_mode multi_token \
  --multi_token_prediction_steps 3 \
  --multi_token_loss_reduction linear \
  --dataset shakespeare_char \
  --out_dir out/multi_token_demo \
  --max_iters 200 \
  --eval_interval 100 \
  --eval_iters 20
