#!/bin/bash

# Demonstration of GNS-driven adaptive batch sizing.
# Uses the SOGNS approximation to monitor the gradient noise scale and
# the sqrt_ratio controller to keep it near a target value.  The batch
# size is increased when the model can benefit from more parallelism and
# decreased when it becomes compute-inefficient.  This helps maximise the
# amount learned per token processed.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
pushd "$script_dir/../" > /dev/null

python train.py \
    --dataset shakespeare_char \
    --block_size 64 \
    --batch_size 4 \
    --max_iters 200 \
    --eval_interval 100 \
    --log_all_metrics \
    --gns_type sogns \
    --gns_variant sqrt_ratio \
    --gns_target 128 \
    --gns_max_batch 512 \
    --gns_ema_beta 0.9

popd > /dev/null
