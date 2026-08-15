# Attention Residual PEFT for Hugging Face models

This experiment freezes a Hugging Face causal LM and trains depth-wise attention
over its residual stream. It currently supports decoder models that expose
`.model.layers` and `.model.norm` (including SmolLM2/Llama), plus GPT-2-style
`.transformer.h` and `.transformer.ln_f` models.

Unlike the repository's full sublayer-level Attention Residual architecture,
this PEFT adapter routes among completed **decoder-layer outputs**. Each layer
and the final norm have a learned query and scalar gate. The gate is initialized
to zero, so attaching the adapter exactly preserves the pretrained function.
Only the queries and gates are optimized; SmolLM2-135M uses 17,887 trainable
parameters (about 0.013% of the base model).

## GSM8K fine-tuning and before/after evaluation

Install PyTorch, Transformers, and Datasets, then run:

```bash
python huggingface_model/attention_residual_peft/finetune_math.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --train_examples 1000 \
  --eval_examples 200 \
  --epochs 1 \
  --batch_size 2 \
  --gradient_accumulation 8 \
  --learning_rate 1e-2 \
  --output_dir out/smollm2-135m-gsm8k-attnres
```

The program evaluates the same deterministic GSM8K test slice before and after
fine-tuning, masks prompt tokens during supervised training, freezes every base
parameter, saves only `attention_residual_adapter.pt`, and writes per-example
predictions plus exact-match summaries to `gsm8k_results.json`.

Evaluate a saved adapter without training:

```bash
python huggingface_model/attention_residual_peft/finetune_math.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --adapter_dir out/smollm2-135m-gsm8k-attnres \
  --output_dir out/smollm2-135m-gsm8k-attnres-eval \
  --eval_examples 200 \
  --eval_only
```

GSM8K exact match is intentionally strict. A credible result should use the full
1,319-example test split, multiple training seeds, and stronger math benchmarks
such as MATH once the pilot establishes a learning signal. Compare against a
parameter-matched LoRA control trained on the identical examples and token
budget. The adapter performs quadratic work in decoder depth and retains earlier
layer outputs, so report memory and throughput as well as accuracy.

## CPU smoke result

The checked-in pilot summary in
`results/smollm2_135m_gsm8k_attention_residual_peft_pilot.json` proves the full
download, before-evaluation, fine-tuning, adapter-save, and after-evaluation path
runs in a CPU-only environment. It is deliberately too small for a quality
claim: eight GSM8K training examples and two test examples produced 0/2 exact
match both before and after. The nonzero learned gates and queries confirm that
the adapter was updated. Use the larger command above to assess quality on a GPU.
