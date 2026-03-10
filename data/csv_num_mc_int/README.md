# CSV Numerical Multicontext (Shift/Scale -> Integer Quantization)

This helper converts a headered CSV of numeric columns into a **scalar numerical multicontext** dataset (uint16 integer tokens) compatible with:

- `--training_mode multicontext`
- `--multicontext`
- `--numerical_multicontext`
- `--numerical_multicontext_input_format scalar`

Each CSV column becomes one context folder. Values are transformed and quantized as:

```text
transformed = (raw + shift) * scale
quantized = clip(round(transformed), clip_min, clip_max)
```

## Files

- `get_datasets.sh`: shell wrapper
- `prepare_csv_int_multicontext.py`: dataset conversion script
- `input.csv`: small template CSV

## Quick start

```bash
data/csv_num_mc_int/get_datasets.sh
```

## Use your own CSV

```bash
data/csv_num_mc_int/get_datasets.sh path/to/input.csv
```

## Per-column shift and scale

```bash
data/csv_num_mc_int/get_datasets.sh path/to/input.csv \
  --output_root csv_num_mc_int_demo \
  --train_ratio 0.9 \
  --column-transform bpm:-40:2 \
  --column-transform spo2:0:10
```

Each context writes:

```text
data/<output_root>/<column>/train.bin
data/<output_root>/<column>/val.bin
data/<output_root>/<column>/meta.pkl
```

`meta.pkl` includes quantization parameters so sampled values can be interpreted or approximately dequantized:

```text
raw ~= (quantized / scale) - shift
```
