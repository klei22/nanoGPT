#!/bin/bash


phonemes=( '5' '_' 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j' 'k' 'm' 'n' 'o' 'p' 'r' 's' 't' 'u' 'w' 'x' 'y' 'z' )

python3 sample2.py --device=cuda --out_dir ./out/20240703_235338_wte_factor-10000-cuda-spaces_and_newlines-True-True-True-True-phoneme_shared_fire_unfrozen_final_really_30deg_cf_fix_5.npy-phoneme_sharedfire_unfrozen_really_30deg_cf_fix_5 --num_samples 1  --show_heatmap --max_new_tokens 1 --chart_type barchart --stats_folder stats

for phoneme in "${phonemes[@]}"; do
  echo "$phoneme"
  python3 sample2.py --device=cuda --out_dir ./out/20240703_235338_wte_factor-10000-cuda-spaces_and_newlines-True-True-True-True-phoneme_shared_fire_unfrozen_final_really_30deg_cf_fix_5.npy-phoneme_sharedfire_unfrozen_really_30deg_cf_fix_5 --num_samples 1  --show_heatmap --start "$phoneme" --max_new_tokens 1 --chart_type barchart --stats_folder stats
done
