# FLORES-200 (haoranxu)

This directory downloads the English-centric FLORES-200 translation test sets published at
[haoranxu/FLORES-200](https://huggingface.co/datasets/haoranxu/FLORES-200).
It focuses on the English pairs for Japanese (`en-ja`), Korean (`en-ko`), and Simplified Chinese (`en-zh`).

## How to download

```bash
cd data/flores-200-haoranxu
bash get_dataset.sh
```

The script pulls the Parquet files for each language pair via `utils/get_parquet_dataset.py`,
writes `#EN:`/`#JA:` (etc.)-prefixed text lines to individual `<pair>.txt` files,
and concatenates them into `input.txt` with a brief `#pair:` separator between groups.
