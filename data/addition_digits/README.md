# Addition digits dataset

This toy dataset builds arithmetic strings such as:

```
1+2=3
42+17=59
```

The language model is trained to predict the digits that appear after the `=`
character and before the newline. The dataset is intentionally small so it can
be generated and tokenized quickly for demo benchmarks.

## Quickstart

```bash
cd data/addition_digits
python prepare.py --max-number 99 --num-samples 20000
```

This command creates the following files in the same directory:

- `addition.txt`: raw text with one addition expression per line.
- `train.bin` / `val.bin`: tokenized splits using a character vocabulary.
- `meta.pkl`: vocabulary metadata for `train.py` and the sampling helpers.

The defaults generate 20k expressions with a 90/10 train/val split. Adjust
`--num-samples` or `--max-number` to control problem difficulty.

## Using the benchmark

After training on this dataset, point `train.py` to the benchmark definition in
`benchmarks/examples/addition.json` to score exact vs. regex-valid answers at
multiple `top_k` values.
