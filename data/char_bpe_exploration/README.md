# FLORES-200 Char-BPE Exploration

This folder contains a reproducible pipeline for comparing nanoGPT's `char_bpe`
tokenizer across selected FLORES-200 languages and stepped vocabulary sizes.

The pipeline does three things:

1. Downloads/loads FLORES-200 via Hugging Face `datasets`.
2. Writes one UTF-8 `.txt` file per requested language in `texts/`.
3. Runs `data/template/prepare.py --method char_bpe` at each requested vocabulary
   size, then writes comparable metrics to `results/summary.csv` and
   `results/summary.json`. By default, 90% of each text becomes `train.bin`
   and 10% becomes `val.bin` so downstream validation-loss comparisons work.

## Languages

| Label | FLORES-200 code |
| --- | --- |
| kiswahili | `swh_Latn` |
| bahasa_indonesian | `ind_Latn` |
| korean | `kor_Hang` |
| korean_nfd | `kor_Hang` converted with `data/hangul/hangul_nfc_to_nfd.py` |
| english | `eng_Latn` |
| chinese | `zho_Hans` |
| japanese | `jpn_Jpan` |
| arabic | `arb_Arab` |
| spanish | `spa_Latn` |
| german | `deu_Latn` |
| russian | `rus_Cyrl` |
| thai | `tha_Thai` |
| filipino | `tgl_Latn` |
| hindi | `hin_Deva` |
| finnish | `fin_Latn` |
| italian | `ita_Latn` |

## Quick start

From the repository root:

```bash
python3 data/char_bpe_exploration/scripts/run_char_bpe_exploration.py
```

Optional examples:

```bash
# Use a smaller sweep while testing the pipeline.
python3 data/char_bpe_exploration/scripts/run_char_bpe_exploration.py \
  --vocab-sizes 384,512

# Refresh language text files from Hugging Face before tokenizing.
python3 data/char_bpe_exploration/scripts/run_char_bpe_exploration.py \
  --refresh-texts

# Use the full text as train.bin when you only need tokenizer compression metrics.
python3 data/char_bpe_exploration/scripts/run_char_bpe_exploration.py \
  --percentage-train 1.0
```

If `datasets` is not installed, install it first:

```bash
python3 -m pip install datasets numpy tqdm sentencepiece tiktoken
```

## Validation-loss and bits-per-byte demo

To train the same small model across language/vocab-size pairs and compare
validation loss plus bits per byte, run:

```bash
bash demos/char_bpe_flores_validation_bpb_demo.sh
```

The demo writes a comparison table to
`out/char_bpe_flores_validation_bpb/summary.csv`. You can override `LANGUAGES`,
`VOCAB_SIZES`, `MAX_ITERS`, `DEVICE`, and other shell variables before running
the script.

## Outputs

Generated artifacts are intentionally ignored by Git:

- `texts/*.txt`: per-language FLORES text files.
- `runs/<language>/vocab_<size>/`: nanoGPT `char_bpe` outputs (`meta.pkl`,
  `train.bin`, `val.bin`, `char_bpe_vocab.json`, token counts, and metrics).
- `results/summary.csv` and `results/summary.json`: comparison table.

Key metrics:

- `bytes_per_token`: lower means the tokenizer compresses that language into
  fewer tokens for the same UTF-8 byte length.
- `chars_per_token`: lower/higher should be interpreted by script and Unicode
  normalization; compare Korean NFC vs NFD carefully because NFD increases the
  number of Unicode code points.
- `unk_byte_fallback_tokens`: count of raw-byte fallback tokens used by the
  trained tokenizer.
