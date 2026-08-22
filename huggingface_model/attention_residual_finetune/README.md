# SmolLM2-135M final attention-residual experiment

This experiment tests the **last** mixture from nanoGPT's Full Attention
Residuals on `HuggingFaceTB/SmolLM2-135M-Instruct`. Every pretrained parameter
is frozen. Forward hooks collect the embedding and decoder-layer outputs, a
single learned pseudo-query applies a token-local softmax over depth, and the
mixture replaces the input to the model's final RMSNorm. The original Hugging
Face model object and its `generate()` method remain intact, which lets the
lm-evaluation-harness run generative benchmarks.

The query is initialized to zero, matching nanoGPT: the initial adapter is an
equal average over all depths and does not reproduce the unmodified model. Only
`final_attention_residual.query` is optimized (576 parameters for SmolLM2-135M).

## Install

Use an environment with PyTorch, then install the experiment dependencies:

```bash
pip install transformers datasets accelerate "lm_eval[math]"
```

## 1. Fine-tune on GSM8K train

```bash
python -m huggingface_model.attention_residual_finetune.train \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --max-steps 500
```

Questions use the model's chat template and loss is applied only to answer
tokens, matching the benchmark's default prompt mode. The script uses only
GSM8K's **train** split and saves
`out/smollm2-135m-final-attention-residual/final_attention_residual.pt`; it does
not save a duplicate copy of the frozen 135M-parameter model.

## 2. Compare before and after

```bash
python -m huggingface_model.attention_residual_finetune.benchmark \
  --adapter out/smollm2-135m-final-attention-residual/final_attention_residual.pt
```

The same model instance is evaluated first without the adapter and then with
the trained adapter on these lm-evaluation-harness tasks:

- `ifeval` for instruction following;
- `gsm8k` for grade-school word problems; and
- `minerva_math` for competition mathematics.

Raw runs are written to `before.json` and `after.json`; the task result tables
are collected in `comparison.json`. For a quick pipeline check before a full
run, add `--limit 10`. A limited run is not a meaningful benchmark result.
The instruct model's chat template is enabled by default for both runs; use
`--no-chat-template` to explicitly compare plain benchmark prompts instead.

GSM8K test examples are evaluated by the harness and are not used by the
training script. Nevertheless, report the GSM8K result as in-domain
fine-tuning, and use IFEval and Minerva Math to inspect transfer and regression.

This adapter currently targets SmolLM2/Llama-style decoder-only models exposing
`model.embed_tokens`, `model.layers`, and `model.norm`.
