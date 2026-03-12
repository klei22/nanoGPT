# Grokking Dataset

This dataset generates modular addition equations intended to reproduce the
classic grokking setup (training on a subset of modular arithmetic pairs and
validating on the remainder).

## Files

- `generate_grokking_data.py` - Generates modular addition examples and splits them.
- `tokens.txt` - Character vocabulary for encoding.
- `prepare.py` - Encodes train/val text into `train.bin`, `val.bin`, and `meta.pkl`.
- `get_dataset.sh` - Runs the full pipeline.

## Usage

```bash
bash get_dataset.sh
```

This produces `train.txt`, `val.txt`, `train.bin`, `val.bin`, and `meta.pkl` in
this folder.
