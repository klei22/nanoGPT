# SYNTH dataset setup

This folder prepares data from the [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) parquet collection. The dataset
contains synthetic query/answer pairs with associated reasoning and metadata.

## Download options

The provided `get_dataset.sh` script wraps the shared parquet helpers in `../template/utils` and supports two modes:

- Default (no flags): download and process only the first two `synth_*.parquet` files to keep the download small.
- Full: pass `--full` (or `-f`) to pull and process every parquet file listed on the dataset page.

Run the script from this directory:

```bash
bash get_dataset.sh         # first two parquet shards only
bash get_dataset.sh --full  # all available shards
```

Both invocations create `downloaded_parquets/`, `json_output/`, and a merged `input.txt` file containing the `query` and
`synthetic_answer` fields (prefixed with `#Q:`/`#A:`) ready for tokenization.
