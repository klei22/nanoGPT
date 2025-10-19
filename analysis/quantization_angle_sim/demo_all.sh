#!/usr/bin/env bash
set -euo pipefail

OUT="quantization_results_sweeps"
mkdir -p "$OUT"
DEVICE="${1:-cpu}"

# 1) Integer fake quant sweeps (int8 -> int3)
echo "[SWEEP] INT 8..3, std=1.0"
python quantize_embedding_sim_stats.py \
  --quantizer int --bits-from 8 --bits-to 3 \
  --std 0.02 --seed 0 \
  --embedding-sizes 128 256 512 1024 2048 \
  --framework torch --dtype float32 --device "$DEVICE" \
  --outdir "$OUT/int_std0.02"

for S in 0.02; do
  for m in 5 4 3 2 1; do
    echo "[SWEEP] EM formats e5m${m} e4m${m} e3m${m} e2m${m}, std=${S}"
    python quantize_embedding_sim_stats.py \
      --quantizer em --em-list "5,${m} 4,${m} 3,${m} 2,${m}" \
      --std "${S}" --seed 0 \
      --embedding-sizes 128 256 512 1024 2048 \
      --framework torch --dtype float32 --device "$DEVICE" \
      --outdir "$OUT/em_std${S}_m${m}"
  done
done

for S in 0.02; do
  for e in 5 4 3 2; do
    echo "[SWEEP] EM formats e${e}m5 e${e}m4 e${e}m3 e${e}m2, std=${S}"
    python quantize_embedding_sim_stats.py \
      --quantizer em --em-list "${e},5 ${e},4 ${e},3 ${e},2" \
      --std "${S}" --seed 0 \
      --embedding-sizes 128 256 512 1024 2048 \
      --framework torch --dtype float32 --device "$DEVICE" \
      --outdir "$OUT/em_std${S}_e${e}"
  done
done

# # 4) (Optional) Try chained mode to simulate progressive quant steps
# echo "[SWEEP] INT chained, std=1.0"
# python quantize_embedding_sim_stats.py \
#   --quantizer int --bits-from 8 --bits-to 3 --chain \
#   --std 1.0 --seed 0 \
#   --embedding-sizes 128 256 512 1024 2048 \
#   --framework torch --dtype float32 --device "$DEVICE" \
#   --outdir "$OUT/int_chain_std1.0"

# echo "[SWEEP] EM chained e4m3->e5m2->e5m3->e6m2, std=1.0"
# python quantize_embedding_sim_stats.py \
#   --quantizer em --em-list "4,3 5,2 5,3 6,2" --chain \
#   --std 1.0 --seed 0 \
#   --embedding-sizes 128 256 512 1024 2048 \
#   --framework torch --dtype float32 --device "$DEVICE" \
#   --outdir "$OUT/em_chain_std1.0"

echo "[DONE] Results in: $OUT/"

