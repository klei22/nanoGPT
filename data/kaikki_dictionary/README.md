# Kaikki.org dictionary dumps

This dataset folder fetches the "All word forms" JSONL dumps published by [Kaikki.org](https://kaikki.org). The downloads mirror the files named `kaikki.org-dictionary-<Language>.jsonl` for the supported languages.

## Download instructions

Run the helper script to pull one or more languages into this folder:

```bash
# Default (English)
bash get_dataset.sh

# Single language
bash get_dataset.sh Korean

# Multiple languages
bash get_dataset.sh English Korean Japanese

# Everything in the curated list
bash get_dataset.sh all
```

Supported languages (verified at Kaikki.org at time of writing): Arabic, Chinese, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Portuguese, Russian, Spanish, and Vietnamese.
