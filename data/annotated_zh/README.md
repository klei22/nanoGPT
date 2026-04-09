# Annotated Chinese OPUS-100 demo

This directory contains a small demo that downloads the OPUS-100 English–
Chinese split and enriches each example with Hanzi metadata.

## Output schema

Each row in the generated JSON array contains:

- `zh`: Original Chinese text from the `en-zh` OPUS-100 split
- `zh_pin`: Pinyin rendering (per-character, space separated)
- `zh_rad1`: Radical-level decomposition tokens for each character
- `zh_rad2`: Graphical/stroke-level decomposition tokens for each character
- `en`: English translation
- `hanzi_metadata`: Full per-character metadata emitted by
  [`annotate_hanzi_metadata.py`](../template/utils/annotate_hanzi_metadata.py)

## Quickstart

```bash
pip install datasets hanzipy
python data/annotated_zh/prepare_annotated_zh.py --limit 200
```

If the OPUS-100 repository requires authentication in your environment, pass a
Hugging Face token via `--hf_token $HF_TOKEN` (or set the `HF_TOKEN`
environment variable).

Use `--split validation` or `--split test` to target other OPUS-100 splits, and
`--limit 0` to process the entire split.

The default output location is `data/annotated_zh/annotated_zh_en.json`.
