#!/bin/bash

python3 phi_format_comparison.py --trials 199 \
  --phi-bits-start 8 --phi-bits-end 12 \
  --int-bits-start 8 --int-bits-end 12 \
  --phi-consts sqrt2,golden_ratio \
  --restricted-phi \
  --grid-steps 257 \
  --refine-steps 257 \
  --grid-span 10 \
  --refine-span 2 \
  --logy \
  --dim 8192

