# Gemma 270M workflows

This folder includes scripts for training Gemma 270M from scratch, fine-tuning, and
experimenting with LM head acceleration using Johnson-Lindenstrauss (JL) projection.

## Loading a pre-trained checkpoint

Use the Hugging Face model hub (or a local checkpoint directory) as the `model_name`:

```bash
python huggingface_model/gemma/270M/jl_head_eval.py \
  --model_name google/gemma-3-270m \
  --dataset_split "train[:1%]" \
  --fineweb_subset sample-10BT
```

The scripts rely on `AutoModelForCausalLM.from_pretrained(...)`, so you can point
`--model_name` at any local checkpoint path created by `train_from_scratch.py` or
`finetune.py`.

## JL-projected LM head evaluation

`jl_head_eval.py` runs a two-stage LM head evaluation:

1. Project hidden states and LM head weights into a lower `target_dimension` using
   a JL random matrix.
2. Compute approximate logits, select the top-`n` candidate tokens, and then compute
   exact logits only for that subset.

It reports validation loss over a fixed number of tokens and can plot a heatmap
for a sweep of `top_n` values and `target_dimension` sizes.

```bash
python huggingface_model/gemma/270M/jl_head_eval.py \
  --model_name google/gemma-3-270m \
  --dataset_split "train[:1%]" \
  --fineweb_subset sample-10BT \
  --eval_tokens 1000 \
  --top_n_values 1000,2000,5000,10000 \
  --target_dimensions 500,400,300,200,100 \
  --output_dir jl_eval_outputs
```
