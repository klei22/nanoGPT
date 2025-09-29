#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python -u resume_from_ckpt.py > "$log" 2>&1
