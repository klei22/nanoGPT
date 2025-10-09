#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python -u infinity_head_search.py > "$log" 2>&1
