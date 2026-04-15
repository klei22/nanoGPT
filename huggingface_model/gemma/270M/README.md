# Gemma 270M workflows

This folder includes scripts for training Gemma 270M from scratch, fine-tuning, and
experimenting with LM head acceleration using Johnson-Lindenstrauss (JL) projection.

## Experiment sequence (English → Spanish)

`finetune.py` now supports the exact experiment structure below.

If you want a single runnable walkthrough, use:

```bash
bash huggingface_model/gemma/270M/demo_gradual_blend_en_es.sh
```

That demo runs the gradual blend path and prints commented commands for the two baselines.

### A) Gradual blend recipe (Softmax → ReLUMax/ReLU2Max + output norm blending)

1. obtain checkpoint
2. optional fine-tune with standard softmax
3. gradual fine-tune with alpha schedule:
   - attention: `alpha * softmax + (1-alpha) * relu_variant`
   - output norms (post-attn + post-ffn): `alpha * output_norm + (1-alpha) * raw`
   - alpha decreases from 1 → 0 and is clamped at 0, with optional `post_zero_steps`

Example:

```bash
# Stage 1: stabilize on Softmax first
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --source_lang en \
  --target_lang es \
  --source_lang_name English \
  --target_lang_name Spanish \
  --dataset_split "train[:10%]" \
  --output_dir ./runs/gemma270_softmax_stage1 \
  --total_iterations 20000 \
  --sample_frequency 1000 \
  --attention_mode softmax

# Stage 2: switch activation and continue from Stage 1 checkpoint
python huggingface_model/gemma/270M/finetune.py \
  --model_name ./runs/gemma270_softmax_stage1 \
  --dataset_config en-es \
  --source_lang en \
  --target_lang es \
  --source_lang_name English \
  --target_lang_name Spanish \
  --dataset_split "train[:10%]" \
  --output_dir ./runs/gemma270_relu2max_stage2 \
  --total_iterations 10000 \
  --sample_frequency 1000 \
  --attention_mode gradual_blend \
  --attention_activation relu2max \
  --activation_divisor 256.0 \
  --alpha_start 1.0 \
  --alpha_end 0.0 \
  --post_zero_steps 1000 \
  --blend_output_norm
```

For `relumax`, only change:

```bash
--attention_activation relumax --activation_divisor 256.0
```

### B) Baseline recipe (Softmax only)

1. obtain checkpoint
2. fine-tune with standard softmax

```bash
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --output_dir ./runs/gemma270_softmax_only \
  --total_iterations 20000 \
  --sample_frequency 1000 \
  --attention_mode softmax
```

### C) Sum baseline (Softmax + ReLU variant scores)

1. obtain checkpoint
2. fine-tune with summed attention probabilities/scores

```bash
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --output_dir ./runs/gemma270_sum_relu2max \
  --total_iterations 20000 \
  --sample_frequency 1000 \
  --attention_mode sum \
  --attention_activation relu2max \
  --activation_divisor 256.0
```

### Important implementation note

The activation swap is implemented by monkey-patching `torch.nn.functional.softmax`
during training, so this is intended as a practical experiment path and not a
production-safe kernel-level replacement.

After each run, the script prints **multi-shot translation outputs for 3 fixed EN→ES
test sentences** (`--print_multishot_after_train` defaults to true).

## Plot validation loss per iteration (Softmax vs ReLUMax vs ReLU2Max)

After running three stage-2 experiments (one each for softmax / relumax / relu2max),
you can plot validation-loss curves from each run's `trainer_state.json`:

```bash
python huggingface_model/gemma/270M/plot_validation_loss.py \
  --run "softmax=./runs/gemma270_softmax_stage2" \
  --run "relumax=./runs/gemma270_relumax_stage2" \
  --run "relu2max=./runs/gemma270_relu2max_stage2" \
  --title "Gemma 270M EN→ES: validation loss per iteration" \
  --output ./runs/gemma270_en_es_val_loss.png
```

## Benchmark EN→ES translation quality

You can run a quick benchmark on a held-out slice with exact-match plus BLEU/chrF
(if `sacrebleu` is installed):

```bash
python huggingface_model/gemma/270M/benchmark_en_es_translation.py \
  --model_name ./runs/gemma270_relu2max_stage2 \
  --dataset_config en-es \
  --dataset_split "train[10%:11%]" \
  --num_samples 200 \
  --max_new_tokens 64
```

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
