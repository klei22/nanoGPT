#!/bin/bash

NSIDE=100

python vector_distribution_analysis.py \
  --format phi -e 2 -m 1 \
  --mode exhaustive \
  --healpix --nside "$NSIDE" \
  --out3d healpix_phi4_e2m1_ns"$NSIDE".html

python vector_distribution_analysis.py  \
  --format fp16 \
  -e 4 -m 3 \
  --mode exhaustive \
  --healpix --nside "$NSIDE" \
  --out3d healpix_e4m3_ns"$NSIDE".html

python vector_distribution_analysis.py  \
  --format fp16 \
  -e 5 -m 2 \
  --mode exhaustive \
  --healpix --nside "$NSIDE" \
  --out3d healpix_e5m2_ns"$NSIDE".html

