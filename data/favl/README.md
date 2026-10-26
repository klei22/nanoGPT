# FAVL shuffled binary datasets

`data/favl/prepare.py` builds frequency-adjusted validation loss (FAVL) dataset
variants from an existing nanoGPT binary dataset folder. The input folder must
contain `meta.pkl`, `train.bin`, and `val.bin`.

The script copies `meta.pkl` unchanged into each output variant and shuffles
`uint16` token order independently inside `train.bin` and `val.bin`. It never
moves tokens between splits, so the token-frequency distribution of each split is
preserved while token order is randomized.

```sh
python data/favl/prepare.py path/to/source_dataset --num_seeds 3 --shuffle_rounds 4
```

By default, variants are written under `data/favl/` as directories named like
`seed_1729_shuffle_rounds_4`. Each variant contains:

- `meta.pkl`
- `train.bin`
- `val.bin`
- `favl_metrics.json`

Use `--seeds 123 456 789` to choose exact seeds, `--output_dir` to place the
variants elsewhere, and `--overwrite` to replace existing variant directories.
