# Benchmarking helpers

This folder contains lightweight evaluation utilities that can run during
`train.py` sampling. The new `benchmarks/inference_benchmarks.py` module
consumes JSON configs describing prompts, expected answers, and scorer types.

## Running inference-time benchmarks

1. Point `train.py` to a config via `--benchmark_config benchmarks/examples/addition.json`.
2. Optionally override runtime parameters:
   - `--benchmark_top_k 5 50` to sweep multiple sampling strategies.
   - `--benchmark_max_new_tokens 8` or `--benchmark_temperature 0.2`.
3. Benchmarks run after each sampling step and log to the console plus
   TensorBoard (if enabled) under `benchmarks/<top_k>/<metric>`.

Each config entry defines:

- `start_tokens`: prompt supplied to the generator.
- `answer_string`: canonical answer for exact matching.
- `legal_answer_regex`: pattern for validating structured outputs.
- `references`: optional list of strings for BLEU scoring.

## Example configs

- `benchmarks/examples/addition.json` pairs with the `data/addition_digits`
  dataset to measure exact vs. regex-valid numeric answers.
- `benchmarks/examples/opus100_translation.json` shows how to attach BLEU and
  exact match scoring for small OPUS-100 translation snippets.

## Quick demo

Run `demos/inference_benchmark_demo.sh` to generate the toy addition dataset,
train a tiny model with the addition benchmark enabled, and then exercise the
OPUS-100 translation benchmark in inference-only mode.
