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

It reports the average absolute token-id difference between the estimated top-k
tokens and the exact top-k tokens, and plots a heatmap for a sweep of `top_n`
values and `target_dimension` sizes.

**How `avg_id_delta` is computed**

For each token position, the script compares the top-`k` token IDs from the
exact logits against the top-`k` token IDs from the JL-approximated logits. It
sorts both ID lists, computes the elementwise absolute difference between the
sorted IDs, and averages those differences across all token positions. It also
tracks how many IDs changed when comparing the two top-`k` sets (order does not
matter), and summarizes that count with the same mean/median/min/max/std stats.

Example (top-`k` = 5):

* Exact top-`k` IDs: `[2, 7, 11, 19, 42]`
* Estimated top-`k` IDs: `[3, 8, 11, 21, 40]`
* Absolute differences: `|2-3|, |7-8|, |11-11|, |19-21|, |42-40|`
* Average: `(1 + 1 + 0 + 2 + 2) / 5 = 1.2`

```bash
python huggingface_model/gemma/270M/jl_head_eval.py \
  --model_name google/gemma-3-270m \
  --dataset_split "train[:1%]" \
  --fineweb_subset sample-10BT \
  --eval_tokens 1000 \
  --top_k 65 \
  --top_n_values 1000,2000,5000,10000 \
  --target_dimensions 500,400,300,200,100 \
  --projection orthonormal \
  --approx_logits_device cpu \
  --approx_logits_dtype float16 \
  --approx_chunk_size 5000 \
  --exact_chunk_size 5000 \
  --annotate_stats true \
  --output_dir jl_eval_outputs
```
