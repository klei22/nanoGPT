# GSM-Symbolic

This directory contains helper scripts for working with the
[GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) dataset.

## Description

GSM-Symbolic pairs GSM8K-style grade school math questions with symbolic
reasoning traces and final numeric answers. The dataset is split across three
JSONL shards (`main`, `p1`, and `p2`) that each expose a `test` split. The
provided `get_dataset.sh` downloads all shards and formats them into
`input.txt` with explicit user (`#U`) and assistant (`#B`) prefixes.

## Usage

```bash
bash get_dataset.sh
```

After running the script, the combined JSON artifacts reside under
`json_output/combined.json` and the training-ready text is written to
`input.txt`.

## Links

- Hugging Face: https://huggingface.co/datasets/apple/GSM-Symbolic
- arXiv: https://arxiv.org/abs/2410.05229

## License

The upstream dataset is released under the CC BY-NC-ND 4.0 license. Please see
`LICENSE-CC-BY-NC-ND-4.0.md` in the source dataset repository for details.
