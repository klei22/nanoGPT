# Dataset Folder README

## Creating a new dataset

To speed up creation of a new dataset, initialize it with the following script.

This will create the necessary directory structure and symlink common utility
scripts for dataset preparation.

```sh
bash create_new_dataset.sh <name-of-dataset>
```

## Combining Datasets

To combine binary dataset files from multiple directories into single train and
validation files, use the `combine_datasets.py` script.

This is useful when you want to merge data from different sources.

```sh
python combine_datasets.py --dirs <list-of-directories> --output_dir <output-directory>
```

## Wishlist

- [ ] Custom phoneme-token-list per language.
- [ ] Script to merge phoneme lists.

## Example: adding haoranxu/flores-200 (en-zh, en-ko, en-ja)

The `data/flores200_haoranxu` folder shows how to adapt the dataset template
for a concrete source. Its `get_dataset.sh` downloads the English–Chinese,
English–Korean, and English–Japanese Parquet shards from
[`haoranxu/flores-200`](https://huggingface.co/datasets/haoranxu/flores-200),
extracts the `en` source text plus the target language, and writes
`input.txt` with `#U:`/`#B:` prefixes ready for tokenization.

To try it out:

```sh
cd data/flores200_haoranxu
pip install huggingface_hub pandas pyarrow  # once per environment
bash get_dataset.sh
```

After the script completes, `input.txt` contains the concatenated three splits
and can be fed to `prepare.py` like any other dataset generated from the
template.

