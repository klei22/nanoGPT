# Gemma 270M English→Indonesian fine-tuning with alternating tokenizers

This guide explains how to fine-tune the Gemma 270M model on OPUS-100 (English ↔ Indonesian)
while gradually introducing an alternate tokenizer (e.g., a lightweight Mistral tokenizer).
It also outlines a publish-ready translation benchmark.

## Alternating tokenization schedule

`huggingface_model/gemma/270M/finetune.py` now supports mixing tokenizers on-the-fly:

* **Primary tokenizer**: Gemma (default: `google/gemma-3-270m`).
* **Alternate tokenizer**: configurable (default: `mistralai/Mistral-7B-Instruct-v0.2`).
* **Ramp**: linearly increase alternate-tokenizer usage from `--alternate_start_ratio`
  to `--alternate_target_ratio` over `--alternate_ramp_steps` steps (defaults to
  `--total_iterations`). This lets you ease in the new tokenizer from 0% to 50% of
  iterations without changing the dataset.
* **Dataset**: OPUS-100 (`en-id` split, first 10% of the train set with a 90/10 split for
  train/eval).

### Example: ramp from 0% → 50%

```bash
python huggingface_model/gemma/270M/finetune.py \
  --total_iterations 3000 \
  --sample_frequency 500 \
  --alternate_tokenizer_name mistralai/Mistral-7B-Instruct-v0.2 \
  --alternate_start_ratio 0.0 \
  --alternate_target_ratio 0.5 \
  --alternate_ramp_steps 3000 \
  --max_length 160
```

Key notes:

1. Tokenization now happens inside the data collator, so no offline preprocessing is needed.
2. The collator picks a tokenizer per batch according to the ramp schedule and masks padding
   tokens with `-100` in the labels.
3. `TokenizerScheduleCallback` keeps the collator in sync with the trainer step counter.

## Publish-grade translation benchmark

For a reproducible, publication-quality assessment, evaluate on a held-out benchmark
(e.g., FLORES-200 or the OPUS-100 validation slice) with both automatic and human-aligned
metrics.

### Automatic metrics

1. **Prepare data**: download the FLORES-200 devtest split for English ↔ Indonesian.
2. **Generate translations** using the fine-tuned checkpoint (primary Gemma tokenizer is
   recommended for inference). Save hypotheses to a text file that aligns one-to-one with
   the references.
3. **Compute metrics**:
   * `sacrebleu` for detokenized BLEU (standardized signature):
     ```bash
     pip install sacrebleu
     sacrebleu reference.id < hypotheses.id > bleu.txt
     ```
   * `COMET` for quality beyond surface overlap:
     ```bash
     pip install unbabel-comet
     comet-score -s source.en -t hypotheses.id -r reference.id --model Unbabel/wmt22-comet-da
     ```
   * Optionally add `chrF` and `TER` via `sacrebleu -m chrf ter` for completeness.

### Human-aligned spot checks

* Sample 100–200 sentences spanning short/long inputs and varied domains.
* Rate adequacy/fluency (Likert 1–5) and compute average scores plus standard deviation.
* Record a few qualitative error examples (mistranslations, dropped named entities,
  hallucinations) to contextualize metric shifts.

### Reporting

Include in the benchmark report:

* Training configuration (tokenizer ramp schedule, dataset slice, max length, steps).
* Checkpoint identifier and decoding parameters (e.g., `max_new_tokens`, `temperature`).
* Automatic metrics with sacrebleu signatures and COMET model version.
* Human evaluation summary plus notable failure cases.
* Comparison against a single-tokenizer baseline to show the effect of the mixed tokenizer
  schedule.
