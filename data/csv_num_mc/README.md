# CSV Numerical Multicontext (FP16 bits)

This dataset helper converts a headered CSV of numeric columns into a **numerical multicontext** dataset compatible with:

- `--numerical_multicontext`
- `--numerical_multicontext_input_format fp16_bits`

Each CSV column becomes one context folder, and values are written as IEEE-754 FP16 bit patterns in `uint16` binary files.

## Files

- `get_datasets.sh`: shell wrapper to run generation
- `prepare_csv_fp16_multicontext.py`: conversion script
- `input.csv`: small template placeholder CSV
- `float_print.py`: helper to inspect binary files as FP16 values

## Input CSV format

- First row must be headers.
- All cells must be numeric.
- All columns must have the same row count.

Example:

```csv
time,temperature,pressure
0,20.0,101.3
1,20.5,101.1
2,21.2,100.9
```

## Quick start

From repo root:

```bash
data/csv_num_mc/get_datasets.sh
```

By default this reads `data/csv_num_mc/input.csv` and writes output under `data/csv_num_mc/`.

## Use your own CSV

```bash
data/csv_num_mc/get_datasets.sh path/to/input.csv
```

## Output location / split ratio

```bash
data/csv_num_mc/get_datasets.sh path/to/input.csv \
  --output_root csv_num_mc_custom \
  --train_ratio 0.9
```

This produces:

```text
data/csv_num_mc_custom/<column_header>/train.bin
data/csv_num_mc_custom/<column_header>/val.bin
data/csv_num_mc_custom/<column_header>/meta.pkl
```

Header names are sanitized into folder names (non alphanumeric chars become `_`).

## Per-column offset and scaling

You can apply a per-column affine transform before fp16 encoding:

```text
transformed = (raw + offset) * scale
```

Pass repeatable `--column-transform` flags in this format:

```text
--column-transform <column>:<offset>:<scale>
```

Example:

```bash
data/csv_num_mc/get_datasets.sh data/csv_num_mc/input.csv \
  --output_root csv_num_mc_demo \
  --train_ratio 0.8 \
  --column-transform temperature:-10:0.1 \
  --column-transform pressure:0:0.01
```

## Metadata

Each context `meta.pkl` includes:

- `numerical_multicontext_input_format: "fp16_bits"`
- `source_column`
- applied `offset` and `scale`
- raw and transformed min/max ranges
- `samples` and `train_ratio`



## Inspect generated `.bin` files

Use `float_print.py` to print values from a `train.bin` or `val.bin` file as `float16`:

```bash
python3 data/csv_num_mc/float_print.py data/csv_num_mc_demo/temperature/train.bin
```

Print with indices:

```bash
python3 data/csv_num_mc/float_print.py data/csv_num_mc_demo/temperature/train.bin --index
```

Read as big-endian data (if needed):

```bash
python3 data/csv_num_mc/float_print.py data/csv_num_mc_demo/temperature/train.bin --endian big
```

> Note: `float_print.py` depends on `numpy`.

