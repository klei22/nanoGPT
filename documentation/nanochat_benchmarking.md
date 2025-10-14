# Benchmarking Insights from `nanochat`

The `nanochat` project ships with a fully scripted evaluation flow that
runs models against the CORE benchmark suite and writes a CSV summary for
each checkpoint.  The entry point is `scripts/base_eval.py`, which loads a
model, iterates over the configured tasks, reports accuracies, and logs a
CORE aggregate score to the run report.  It also supports evaluating
Hugging Face checkpoints directly and captures per-task timing to aid
regression tracking.【F:nanochat/scripts/base_eval.py†L13-L117】【F:nanochat/scripts/base_eval.py†L129-L195】

The underlying scoring utilities in `nanochat/nanochat/core_eval.py`
handle prompt rendering for multiple task formats, batching sequences for
efficient inference, and computing per-example correctness.  The module
implements evaluators for multiple-choice, schema, and language modeling
setups, including careful prefix/suffix handling so that loss windows
align with continuation spans.【F:nanochat/nanochat/core_eval.py†L1-L165】【F:nanochat/nanochat/core_eval.py†L167-L261】

Although the main `nanoGPT` training path already exposes an `lm_eval`
wrapper through `sample.py`, it lacked a dedicated CLI for running a
standard benchmark sweep and emitting summary artefacts tied to a
checkpointed run.  This made it harder to compare models or wire the
results into automated pipelines.

## Bridging the Gap in `nanoGPT`

To bring the `nanochat` benchmarking workflow into the more general
`nanoGPT` repository, this change introduces `benchmarks/run_lm_eval.py`.
The script mirrors `nanochat`'s ergonomics: it accepts a training run
(directory plus checkpoint name), automatically restores the model and
associated tokenizer metadata, and executes a configurable lm-eval task
suite via the existing `NanoGPTLM` adapter.  Results are written to a
JSON file, a lightweight summary is stored alongside the run, and an
optional CSV mirrors the per-task table seen in `nanochat`.

Key elements include:

- Model restoration from `ckpt.pt` with dropout disabled, matching the
  inference setup used by `sample.py`.
- Tokenizer resolution that looks for `meta.pkl` files in the run
  directory or dataset tree, falling back to a Hugging Face tokenizer for
  GPT-2 variations.
- Rich-powered console tables to echo the benchmark summary in the
  terminal while still emitting machine-readable artefacts for automation.

The supporting `utils/tokenizer_utils.py` module centralises the encode
and decode helpers that were previously defined inside `sample.py`.  This
allows the training script, sampling CLI, and the new benchmarking tool
to share tokenizer logic without circular imports.

## Using the new benchmark CLI

```
python benchmarks/run_lm_eval.py --out_dir out --ckpt_name ckpt.pt \
    --tasks arc_challenge,arc_easy,gsm8k,mmlu,humaneval \
    --eval_batch_size 4 --summary_csv out/benchmarks/latest.csv
```

The command loads `out/ckpt.pt`, locates its tokenizer metadata,
runs the lm-eval tasks, prints a summary table, and saves raw/summary
JSON files under `out/benchmarks/`.  Passing `--summary_csv` produces a
spreadsheet-compatible snapshot for downstream dashboards.
