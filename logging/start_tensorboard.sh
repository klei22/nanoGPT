#!/bin/bash
# Note: may need to run this script as `source start_tensorboard.sh` to utilize environment

tensorboard --logdir=./logs || python3 -m tensorboard.main --logdir=./logs
