# Premade vocab sets

This directory stores pre-extracted token lists for use with the JSON byte-fallback
pipeline. Each file is a JSON array of string tokens (UTF-8), matching the format
consumed by `prepare.py --method json_byte_fallback`.

## Current vocab sets (largest to smallest)

| File | Source model | Size (bytes) |
| --- | --- | --- |
| `gemma_tokens.json` | `google/gemma-2b` | 3,218,566 |
| `deepseek_tokens.json` | `deepseek-ai/deepseek-llm-7b-base` | 415,577 |

## Adding additional vocabularies

Use `download_vocab_set.py` to fetch a tokenizer from Hugging Face, remove byte
fallback tokens, and write the normalized JSON list.

```bash
python download_vocab_set.py \
  --model google/gemma-2b \
  --output gemma_tokens.json

python download_vocab_set.py \
  --model deepseek-ai/deepseek-llm-7b-base \
  --output deepseek_tokens.json
```

### Example: Qwen (and other models)

```bash
python download_vocab_set.py \
  --model Qwen/Qwen2.5-7B \
  --output qwen2_5_tokens.json
```

Notes:
- The script attempts `tokenizer.json` first, then `tokenizer.model`. The latter
  requires `sentencepiece` to be installed.
- Byte fallback tokens (single-byte entries and `<0xNN>` markers) are removed by
  default; pass `--keep-byte-tokens` if you need to preserve them.
