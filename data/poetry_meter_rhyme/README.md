# Poetry meter and rhyme

This dataset builder creates explicit poem/author-level training, validation, and
gold benchmark splits for meter, scansion, and rhyme-scheme tasks. Generated data
and downloaded source annotations are deliberately not committed.

## Sources and reuse status

* [Metrical Tagging in the Wild](https://github.com/tnhaider/metrical-tagging-in-the-wild)
  supplies English gold syllable stress, meter, measure, and scansion with its
  published train/dev/test division. The poems are public domain, but the
  annotation repository does not state reusable annotation terms.
* [Chicago rhyme data](https://github.com/sravanareddy/rhymedata) supplies gold
  poem rhyme classes. Its underlying poems are public domain, but annotation
  reuse terms are not explicit.
* [Deep-speare](https://github.com/jhlau/deepspeare) is optional, training-only
  background material. It is disabled unless `--include-background` is given.

The first two downloads are therefore gated and intended for local research use.
This repository does **not** redistribute their normalized annotations. Confirm
reuse terms before publishing generated files. Each URL, commit, checksum, and
citation is pinned in `sources.lock.json`.

## Build and validate

```bash
./get_dataset.sh --accept-research-only
python3 prepare.py -t generated/train.txt -v generated/val.txt \
  --method tiktoken --tiktoken_encoding gpt2 \
  --additional_tokens_file special_tokens.json --track_token_counts
```

`get_dataset.sh` writes only training examples to `input.txt`; benchmark poems
remain in `generated/test.jsonl`. The builder preserves Haider's supplied split,
splits Chicago by poet, canonicalizes rhyme classes, removes normalized duplicate
lines/poems, strips title/author benchmark cues, and emits `manifest.json` with
counts, rejects, checksums, and leakage results.

Evaluate a checkpoint from the repository root:

```bash
python3 benchmarks/evaluate_meter_rhyme.py \
  --ckpt_path out/poetry_meter_rhyme/ckpt.pt \
  --out_dir out/poetry_meter_rhyme \
  --benchmark_file data/poetry_meter_rhyme/generated/test.jsonl \
  --tasks meter_minimal_pair,rhyme_minimal_pair,joint_repair \
  --device cuda --dtype bfloat16 --length_norm \
  --output_json out/poetry_meter_rhyme/meter_rhyme_final.json
```

## End-to-end demo and exploration

The demo performs the gated build, explicit-split tokenization, compact training,
and evaluation of every periodic checkpoint so that meter/rhyme learning curves
can be plotted from the JSON results:

```bash
demos/poetry_meter_rhyme_benchmark.sh --accept-research-only
```

Pass `--help` for CPU, data-only, existing-checkpoint, output-directory, and
skip-stage options. For a baseline-versus-rotary/QK-normalization comparison, use
`explorations/poetry_meter_rhyme_benchmark.yaml` after preparing the dataset.
