# ntrex dataset setup

This folder documents how to pull the multilingual `ntrex` parquet files convert
them into a tokenized dataset that matches the rest of the repository.

## 1) Gather the raw parquet files

Use the provided `get_dataset.sh` script to download the parquet shards and flatten the fields you care about into a single `input.txt` file. The script defaults to the dataset homepage URL on Hugging Face and uses the shared parquet helper from `data/template/utils`.

```bash
bash get_dataset.sh
```

The script currently pulls three English-like columns (`eng_Latn`, `kor_Hang`, and `zho_Hans`) and prefixes each row with `#EN:\n` to keep splits visible in the final text file. Adjust the `include_keys`/`value_prefix` arrays in the script if you want to extract other languages from the parquet schema. The helper will download the parquet files, convert each to JSON, then emit the requested keys line-by-line into `input.txt`.

## 2) Tokenize the text

After `input.txt` is created, run the standard template tokenizer to build train/validation binaries. You can switch methods or vocab size depending on your needs.

```bash
python3 ../template/prepare.py -t input.txt --method sentencepiece --vocab_size 32000
```

This produces `train.bin`, `val.bin`, and `meta.pkl` in the current directory for use with `train.py` and related scripts. If you prefer TikToken or character-level tokenization, pass `--method tiktoken` or `--method char` instead.

## 3) Re-run after changes

If you tweak the include keys or add more parquet files, delete `input.txt` (and optionally the `downloaded_parquets` and `json_output` helper folders) before rerunning `get_dataset.sh` so the regenerated text reflects your changes.
