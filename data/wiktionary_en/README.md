# English Wiktionary dataset

This folder downloads English Wiktionary data produced by [Wiktextract](https://github.com/tatuylonen/wiktextract) and turns it into `input.txt` entries that combine the headword, part of speech, glosses, and usage examples.

The default source (`https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz`) contains the full English-language Wiktionary dump, which is large (several gigabytes). For faster experimentation, you can point `--download_url` at a smaller derivative such as the Simple English extract at `https://kaikki.org/dictionary/downloads/simple/simple-extract.jsonl.gz`.

## Usage

```bash
python get_dataset.py \
  --output_file input.txt \
  --download_url https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz
```

Optional flags:

- `--max_entries` stops after writing the requested number of senses (useful for quick smoke tests).
- `--language` chooses the `lang` field to keep (defaults to `English`).
- `--compressed_path` overrides where the downloaded dump is stored.

After generating `input.txt`, run `prepare.py` with your preferred tokenization strategy.
