# SYNTH Dataset

This folder prepares a text dump from the [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) parquet collection.

- `get_dataset.sh` downloads the first two parquet shards by default so you can quickly inspect a slice of the data.
- Pass `--full` (or `--all`) to download and process every shard listed on the dataset page.
- Output is written to `input.txt`, with each entry formatted as a Q/A pair that also includes the model's reasoning.
