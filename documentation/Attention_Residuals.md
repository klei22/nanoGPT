# Attention Residuals

Set `attention_residual_variant: full` (or pass
`--attention_residual_variant full`) to replace the running residual sum with
token-local attention over depth.

For each Transformer block, the implementation:

1. mixes the embedding and earlier sublayer outputs for the attention input;
2. appends the raw self-attention output to depth memory;
3. computes a separate mixture for the MLP input; and
4. appends the raw MLP output.

One final mixture is passed to `ln_f`. Each destination owns a learned
pseudo-query. Queries are initialized to zero, so every mixture initially is an
equal-weight average. Keys are parameter-free RMS-normalized source vectors,
values are the raw vectors, and softmax is only over depth. Consequently this
feature does not replace causal token-to-token self-attention.

Full Attention Residuals store the embedding and all `2 * n_layer` sublayer
outputs, and perform quadratic work in the number of sublayers. The current
implementation supports sequential attention-then-MLP blocks without post-LN;
the usual PreNorm configuration is supported. Use `standard` (the default) for
the existing additive residual architecture and checkpoint compatibility.

## Warm-starting from a Hugging Face checkpoint

The pretrained-weight importer currently supports the Hugging Face GPT-2 family
(`gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl`). It copies the pretrained
embeddings, attention, MLP, normalization, and LM-head weights. Depth-routing
queries do not exist in the source checkpoint, so they remain newly initialized
and must be trained.

This full sublayer-level variant is an **architecture-changing full fine-tune**,
not a normal LoRA run. A generic `AutoModelForCausalLM` cannot be switched to it
by setting a config field: each model family's forward pass must expose raw
attention and MLP outputs. For a function-preserving, layer-level PEFT
approximation that works with SmolLM2/Llama and GPT-2 Hugging Face models, see
[`huggingface_model/attention_residual_peft`](../huggingface_model/attention_residual_peft/README.md).
Do not interpret a tokenizer or weight-conversion mismatch as an architectural
result.

### Controlled before/after experiment

Use three checkpoints rather than only two:

1. **Before:** the untouched Hugging Face checkpoint.
2. **Control:** the same checkpoint fine-tuned with ordinary residual addition.
3. **Treatment:** the same checkpoint and training recipe fine-tuned with Full
   Attention Residuals.

The control separates gains caused by the data and extra optimization from gains
caused by the residual architecture. Keep dataset bytes, tokenizer, sample order,
seed, effective batch size, token budget, optimizer, learning-rate schedule,
precision, and evaluation settings identical. Match *trained tokens*, not just
steps, and record peak memory and wall-clock time because Full Attention
Residuals retain every sublayer output and add depth-wise mixing work.

Prepare a representative causal-language-modeling corpus using the GPT-2
tokenizer. For example, after placing text in `train.txt` and `validation.txt`:

```bash
python data/minipile/prepare.py \
  --train_input train.txt \
  --val_input validation.txt \
  --method tiktoken \
  --tiktoken_encoding gpt2
```

Run the additive control and treatment from the same pretrained model. These are
pilot settings, not recommended final hyperparameters; use a token budget large
enough for validation loss to stabilize.

```bash
COMMON_ARGS=(
  --init_from gpt2
  --gpt2_type gpt2
  --dataset minipile
  --device cuda:0
  --dtype bfloat16
  --batch_size 8
  --gradient_accumulation_steps 8
  --block_size 1024
  --max_iters 5000
  --learning_rate 2e-5
  --min_lr 2e-6
  --decay_lr
  --warmup_iters 200
  --eval_interval 250
  --eval_iters 200
)

python train.py "${COMMON_ARGS[@]}" \
  --attention_residual_variant standard \
  --out_dir out/gpt2-additive

python train.py "${COMMON_ARGS[@]}" \
  --attention_residual_variant full \
  --out_dir out/gpt2-attnres
```

Zero routing queries initially produce a uniform average of available depth
sources, not the pretrained model's cumulative residual sum. Expect an immediate
loss discontinuity in the treatment. Before a costly run, perform a short
learning-rate sweep and plot held-out loss during adaptation. A useful extra
ablation is to freeze imported weights briefly and train only routing queries.
The Hugging Face PEFT implementation linked above does this automatically and
adds zero-initialized gates so attachment preserves the pretrained function.

### Benchmarking

Score the untouched model with the Hugging Face evaluator, then score both
nanoGPT checkpoints with the custom-model evaluator. Both evaluators share the
multiple-choice extraction and continuation-likelihood scoring code.

```bash
BENCHMARKS=hellaswag,arc-easy,arc-challenge,sciq,piqa,winogrande,boolq
mkdir -p results

python benchmarks/evaluate_huggingface_models.py \
  --model_name gpt2 \
  --benchmarks "$BENCHMARKS" \
  --device cuda \
  --dtype bfloat16 \
  --split validation \
  --output_json results/gpt2-before.json

python benchmarks/evaluate_custom_models.py \
  --out_dir out/gpt2-additive \
  --benchmarks "$BENCHMARKS" \
  --device cuda \
  --dtype bfloat16 \
  --split validation \
  --output_json results/gpt2-additive.json

python benchmarks/evaluate_custom_models.py \
  --out_dir out/gpt2-attnres \
  --benchmarks "$BENCHMARKS" \
  --device cuda \
  --dtype bfloat16 \
  --split validation \
  --output_json results/gpt2-attnres.json
```

Use `--max_examples 100` only for a pipeline smoke test. For reported results,
evaluate the complete labeled validation split, preserve the length-normalization
setting across runs, and include validation perplexity as well as downstream
accuracy. Run at least three training seeds and report the mean and spread of the
per-seed treatment-minus-control delta. If possible, use a paired bootstrap over
benchmark examples; a tiny aggregate accuracy increase without an uncertainty
interval is not evidence of improvement.

The primary comparison is **treatment versus fine-tuned additive control**. The
untouched before score answers whether the complete fine-tuning procedure helped;
it does not by itself identify the effect of Attention Residuals. Also report
parameter count, trained tokens, training FLOPs or elapsed time, and peak memory
so a quality gain can be weighed against the architecture's cost.
