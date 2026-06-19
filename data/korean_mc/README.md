# Korean multicontext Hangul factor dataset

This dataset layout is intended for English-to-Korean experiments using the Korean side of OPUS-100 English-Korean parallel data.

## Source and license

`get_dataset.sh` downloads OPUS-100 English-to-Korean data and extracts the Korean target text into `input.txt`. OPUS-100 is a multilingual parallel corpus derived from OPUS collections; users should review the OPUS-100 dataset card and the licenses of the underlying OPUS corpora before redistribution or commercial use.

## Pipeline

1. Download English-Korean OPUS-100 parallel data.
2. Extract Korean target segments into `input.txt`.
3. Run `../template/utils/korean/extract_multicontext_streams.py input.txt .`.
4. The extractor writes one aligned `input.txt` stream for each of the 23 Hangul factor lanes plus `char/input.txt` containing the original character stream.
5. Run `../template/prepare.py --method char -s -S <lane_name>` for every lane directory so nanoGPT can train with `--training_mode multicontext --multicontext`.

Non-Hangul characters are preserved in `char/input.txt` and the metadata sidecars. Their feature lanes use `NON_HANGUL` in the `script` lane and `PAD` markers elsewhere.
