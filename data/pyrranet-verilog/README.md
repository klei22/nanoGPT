# PyraNet-Verilog

This folder contains helpers for working with the [PyraNet-Verilog](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog) dataset from Hugging Face. The dataset contains Verilog code samples along with structured descriptions, rankings, and compilation metadata.

## Download + extract

Use `get_dataset.sh` to download the dataset with `datasets` and extract each Verilog program into its own file.

```bash
bash data/pyrranet-verilog/get_dataset.sh
```

By default the files are written to:

```
data/pyrranet-verilog/orig/orig_<index>.txt
```

### Options

The underlying `prepare.py` script supports common options:

```bash
python data/pyrranet-verilog/prepare.py --help
```

Notable flags:

- `--output-dir`: output directory for extracted files.
- `--max-rows`: limit the number of rows for a smaller sample.
- `--no-streaming`: download the full dataset before processing (large).
- `--overwrite`: rewrite existing files.

## License

PyraNet-Verilog is licensed under **CC BY-NC-SA 4.0**. Refer to the dataset card for full terms and ensure your usage complies with the license.
