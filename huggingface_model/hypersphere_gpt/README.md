# Hypersphere GPT Hugging Face Workflow

This directory contains a Hugging Face compatible implementation of the Hypersphere GPT
architecture together with utilities for training on FineWeb, exporting checkpoints,
running generation, and evaluating on the lm-evaluation-harness benchmark suites.

## Normalization options

The model can be instantiated with a selectable normalization strategy via
`--norm_type` on the training script (and stored in the configuration for reuse
at inference/export time). The following options mirror the implementations in
[`variations/norm_variations.py`](../../variations/norm_variations.py):

| `--norm_type` value | Description |
| --- | --- |
| `layernorm` | Standard LayerNorm without a bias vector. |
| `rmsnorm` | Bias-free RMSNorm. |
| `hypersphere` | HyperSphereNorm with a fixed radius (defaults to √embedding dim). |
| `hypersphere_learned_radius` | HyperSphereNorm with a learned radius parameter. |
| `prmsnorm` | Partial RMSNorm that normalizes over the first `--prmsnorm_pct` fraction of features. |

When using one of the HyperSphereNorm variants you may optionally pin the radius
with `--hsnorm_radius <float>`. For pRMSNorm you can choose how much of the
channel dimension participates in the RMS calculation by supplying
`--prmsnorm_pct` (default: `0.5`).

## Training on FineWeb-Edu

The hero configuration (~124M parameters) is wired into
`train_hypersphere_gpt.py`. A minimal launch that trains with RMSNorm and saves a
checkpoint every 200 steps looks like:

```bash
python huggingface_model/hypersphere_gpt/train_hypersphere_gpt.py \
  --output_dir runs/hypersphere-gpt-rmsnorm \
  --dataset HuggingFaceH4/fineweb-edu \
  --dataset_split "train[:0.1%]" \
  --block_size 2048 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 1 \
  --learning_rate 3e-4 \
  --save_steps 200 \
  --norm_type rmsnorm
```

### Sample training → sampling → benchmarking flow (RMSNorm)

```bash
# 1. Train the model with RMSNorm
python huggingface_model/hypersphere_gpt/train_hypersphere_gpt.py \
  --output_dir runs/rmsnorm-hero \
  --dataset HuggingFaceH4/fineweb-edu \
  --dataset_split "train[:0.1%]" \
  --norm_type rmsnorm

# 2. Generate text from the latest checkpoint
python huggingface_model/hypersphere_gpt/infer_hypersphere_gpt.py \
  --model_path runs/rmsnorm-hero \
  --prompt "The future of open language models"

# 3. Benchmark with the lm-evaluation-harness defaults
python huggingface_model/hypersphere_gpt/run_hypersphere_gpt_benchmarks.py \
  --model_path runs/rmsnorm-hero \
  --tokenizer_name gpt2
```

## Inference-only usage

Once you have a directory containing a trained checkpoint, invoke
`infer_hypersphere_gpt.py` to sample text:

```bash
python huggingface_model/hypersphere_gpt/infer_hypersphere_gpt.py \
  --model_path runs/hypersphere-gpt-rmsnorm \
  --prompt "Education data reveals" \
  --max_new_tokens 128 \
  --temperature 0.8
```

## Exporting to the Hugging Face Hub

Convert a local checkpoint into a standard Hugging Face folder (optionally
pushing to the Hub) with:

```bash
python huggingface_model/hypersphere_gpt/export_hypersphere_gpt_to_hf.py \
  --checkpoint_dir runs/hypersphere-gpt-rmsnorm \
  --output_dir export/hypersphere-gpt-rmsnorm \
  --use_safetensors
```

Pass `--push_to_hub`, `--hub_model_id`, and `--hub_token` to publish the
artifacts directly.

## Running lm-eval benchmark suites

`run_hypersphere_gpt_benchmarks.py` wraps lm-evaluation-harness. The default
invocation benchmarks on a suite of standard language understanding tasks:

```bash
python huggingface_model/hypersphere_gpt/run_hypersphere_gpt_benchmarks.py \
  --model_path export/hypersphere-gpt-rmsnorm \
  --tokenizer_name gpt2 \
  --tasks hellaswag,arc_easy,arc_challenge,winogrande,lambada_openai
```

Use `--device cpu` if you do not have access to a GPU, and set `--dtype float32`
when running entirely on the CPU.
