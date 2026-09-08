# RMSNorm low-gain channel zeroing

This experiment freezes a Hugging Face causal language model, finds the
lowest-magnitude per-channel gains in every RMSNorm layer, and optimizes only
those entries. Training minimizes the usual next-token loss plus an L1 penalty
on the selected gains. At the end, selected gains below `--zero-threshold` are
set to exactly zero. A gradient mask and an after-step restore keep every
unselected parameter bit-for-bit unchanged.

The experiment is intended to answer whether channels which already have small
normalization gains can be removed on a strategically chosen distribution with
little or no language-modeling loss. It is not a general fine-tuning recipe.

## Dataset presets

`--dataset` accepts a Hugging Face dataset name, while `--dataset-config` and
`--text-column` handle its schema. The following starting points target
different capabilities:

| Capability | Arguments |
| --- | --- |
| Broad language | `--dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --text-column text` |
| Code | `--dataset codeparrot/codeparrot-clean --text-column content` |
| Mathematical reasoning | `--dataset openai/gsm8k --dataset-config main --text-column question --extra-text-column answer` |

Review dataset licenses and gated-access requirements before running an
experiment. Dataset rows are streamed by default, so a run does not need to
download a full corpus.

## Example

From the repository root:

```bash
python huggingface_model/finetuning_experiments/rmsnorm_channel_zeroing/finetune.py \
  --model google/gemma-3-270m \
  --dataset openai/gsm8k \
  --dataset-config main \
  --text-column question \
  --extra-text-column answer \
  --selection-fraction 0.05 \
  --zero-penalty 0.1 \
  --max-steps 500 \
  --output-dir outputs/gemma-rmsnorm-zero-gsm8k
```

Use `--dry-run` to print the selected RMSNorm layers and channels without
loading a dataset or training. The output directory contains the model,
tokenizer, `selection.json`, and `experiment_metrics.json`. Compare the initial
and final evaluation losses in the metrics file, and repeat with several seeds
and datasets before drawing conclusions.

## Smoke test

The unit tests exercise selection, gradient masking, restoration, and exact
thresholding without downloading a model or dataset:

```bash
python -m unittest discover \
  -s huggingface_model/finetuning_experiments/rmsnorm_channel_zeroing/tests \
  -v
```
