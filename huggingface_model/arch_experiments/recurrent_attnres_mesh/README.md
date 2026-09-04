# Recurrent Attention-Residual Mesh

A standalone Hugging Face/PyTorch research implementation of a recurrent
all-module mesh, configured for architecture ablations on one NVIDIA A100
80 GB.

Instead of a conventional sequential Transformer stack, the model maintains
parallel attention and feed-forward branches. Every branch receives the
original token embedding directly at the first step. At every later recurrent
step, each destination branch independently routes over the original embedding
and all attention/FFN outputs from the preceding step. All branches update
synchronously. A final Attention-Residuals-style router combines the embedding
and terminal branch states before RMSNorm and the tied language-model head.

## Architecture

Let `E` be the input embedding and `H_i^(t)` be branch `i` at recurrent step
`t`:

```text
H_i^(0) = Module_i(E)

H_i^(t) = Module_i(
    Router_i([E, H_1^(t-1), ..., H_M^(t-1)])
)

logits = LMHead(RMSNorm(
    OutputRouter([E, H_1^(T), ..., H_M^(T)])
))
```

The default `attnres` router uses destination-specific learned pseudo-queries,
RMS-normalized source keys, unnormalized values, and softmax attention across
the embedding/module axis.

The hardware-oriented implementation:

- Packs all attention weights and all FFN weights into batched tensors.
- Folds the attention-module axis into the batch for one causal SDPA call.
- Executes the FFN modules with batched matrix multiplications.
- Uses fixed-length packed batches so Flash-SDPA needs no padding mask.
- Supports BF16, TF32, fused AdamW, `torch.compile`, and recurrent checkpoint
  chunks.
- Pads the vocabulary to a Tensor-Core-friendly multiple.
- Uses expandable CUDA allocator segments to reduce fragmentation.

This package is optimized for training experiments. Generation works without a
KV cache and is therefore not optimized for inference latency.

## Package contents

```text
recurrent_attnres_mesh/
├── README.md
├── requirements.txt
├── recurrent_attnres_mesh.py
└── scripts/
    ├── _common.sh
    ├── setup_env.sh
    ├── check_install.sh
    ├── run_inspect.sh
    ├── run_smoke.sh
    ├── run_benchmark_a100.sh
    ├── run_train_a100.sh
    ├── run_sweep_a100.sh
    ├── run_untied_sweep_a100.sh
    └── run_branch_eval_matched_a100.sh
```

Every run launcher forwards additional CLI arguments supplied after its own
name, so any model or training setting can be overridden without editing the
shell script.

## Requirements and installation

Primary target:

- One full NVIDIA A100 with at least 75 GiB visible HBM
- Compute capability 8.0
- Python 3.10 or newer
- A current NVIDIA driver compatible with the selected PyTorch CUDA build
- Storage for Hugging Face data, checkpoints, and compiled kernels

From the extracted project directory:

```bash
./scripts/setup_env.sh
source .venv/bin/activate
./scripts/check_install.sh
```

The setup launcher defaults to the validated PyTorch 2.5.1 CUDA 12.4 wheel.
If your driver or cluster requires a different official wheel channel:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX \
TORCH_VERSION=X.Y.Z \
  ./scripts/setup_env.sh
```

Replace `cuXXX` with the wheel channel appropriate for your driver and PyTorch
installation. You can also install the dependencies manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `a100-80gb` profile performs a strict preflight before allocating the
model. It rejects CPU execution, non-A100 devices, 40 GB cards or small MIG
slices, compute capabilities other than SM80, multi-process launches, missing
BF16 support, insufficient free-memory reserve, and an unusable Flash-SDPA
head shape/dtype. This prevents a benchmark from silently falling back to a
different kernel or device.

## Default A100 80 GB profile

| Setting | Default |
|---|---:|
| Hidden size | 2048 |
| Attention heads / head dimension | 32 / 64 |
| SwiGLU width | 5504 |
| Attention modules | 4 |
| FFN modules | 4 |
| Recurrent steps | 6 |
| Iteration weights | Tied |
| Sequence length | 2048 |
| Microbatch / gradient accumulation | 8 / 2 |
| Tokens per optimizer update | 32,768 |
| Exact parameter estimate | 305,434,624 |
| Precision | BF16 with TF32 enabled |
| Flash-SDPA | Required and probed |
| Gradient checkpointing | Enabled, two-step chunks |
| `torch.compile` | Enabled, `default` mode |
| HBM target / reserve | 90% / 8 GiB |

Explicit CLI values and isolated sweep run-config values override these profile
defaults. Use `--hardware-profile portable` for CPU or non-A100 checks.

## First run

Inspect the resolved profile without allocating the model:

```bash
./scripts/run_inspect.sh
```

Run the comprehensive tiny-model correctness test:

```bash
./scripts/run_smoke.sh
```

The smoke suite covers forward/backward, causal and padding behavior, finite
router gradients, checkpoint equivalence, attention-only and FFN-only models,
tied and untied recurrence, router controls, padded vocabulary handling,
generation, resizing, and fresh-process Hugging Face `AutoModel` loading.

Next, run the full synthetic training-step benchmark on the A100:

```bash
./scripts/run_benchmark_a100.sh
```

Short capacity probe:

```bash
BENCHMARK_WARMUP=1 BENCHMARK_STEPS=1 \
  ./scripts/run_benchmark_a100.sh
```

The benchmark includes forward, cross-entropy, backward, gradient accumulation,
gradient clipping, and the optimizer update. It reports compiler/warmup and
steady-state memory separately, as well as the capacity peak.

Important result fields include:

- `training_tokens_per_second`
- `estimated_training_tflops`
- `estimated_a100_bf16_mfu`
- `capacity_peak_allocated_gb`
- `capacity_peak_reserved_gb`
- `peak_reserved_fraction`
- `within_memory_budget`
- Flash-SDPA probe status and dtype
- Exact parameter and module-evaluation counts

The MFU-like value includes checkpoint recomputation FLOPs, so it is closer to
an estimated hardware utilization than the narrowest conventional model-FLOP
utilization definition.

## Train on TinyStories

The launcher defaults to `roneneldan/TinyStories`, the GPT-2 tokenizer, 1,000
optimizer steps, and a unique fresh output directory:

```bash
./scripts/run_train_a100.sh
```

Configuration through environment variables:

```bash
DATASET=roneneldan/TinyStories \
TOKENIZER=gpt2 \
MAX_STEPS=5000 \
OUTPUT_DIR="$PWD/runs/tinystories_5k" \
  ./scripts/run_train_a100.sh \
  --save-interval 500
```

For reproducible Hub inputs, pin revisions:

```bash
DATASET_REVISION=<commit-sha> \
TOKENIZER_REVISION=<commit-sha> \
  ./scripts/run_train_a100.sh
```

Each training run requires a new or empty output directory, preventing stale
metrics or checkpoints from being mistaken for a new experiment.

Local text data is also supported:

```bash
./scripts/run_train_a100.sh \
  --train-file data/train.txt \
  --eval-file data/validation.txt \
  --output-dir runs/local_data
```

## Ablation launchers

### Tied module/step grid

```bash
MAX_STEPS=1000 ./scripts/run_sweep_a100.sh
```

The default 45-run grid is:

- Attention modules: `1,2,4`
- FFN modules: `1,2,4`
- Recurrent steps: `1,2,4,6,8`
- Router: `attnres`
- Iteration weights: tied

It uses microbatch 4 with accumulation 4, preserving the profile's 32,768
tokens per optimizer update with more HBM headroom across the grid.

For the 75-run publication grid, include zero-module axes so the sweep also
contains attention-only and FFN-only controls (the invalid `A=0,F=0` cases are
filtered automatically):

```bash
FULL_GRID=1 MAX_STEPS=1000 ./scripts/run_sweep_a100.sh
```

Preview without training:

```bash
./scripts/run_sweep_a100.sh \
  --sweep-dry-run \
  --sweep-max-runs 10
```

Override the axes with environment variables:

```bash
ATTENTION_MODULES=1,2 \
FFN_MODULES=1,2,4 \
ITERATIONS=2,4,8 \
  ./scripts/run_sweep_a100.sh
```

### Memory-safer untied grid

```bash
MAX_STEPS=1000 ./scripts/run_untied_sweep_a100.sh
```

This launcher uses microbatch 2 and accumulation 8, retaining 32,768 tokens per
update while leaving more HBM for weights, gradients, and optimizer states that
grow with untied recurrent depth.

### Branch-evaluation-matched comparison

```bash
MAX_STEPS=1000 ./scripts/run_branch_eval_matched_a100.sh
```

It sequentially runs `(A=1,F=1,S=8)`, `(A=2,F=2,S=4)`, and
`(A=4,F=4,S=2)`. Each point executes eight attention branches and eight FFN
branches per token, helping separate topology from branch-evaluation count.
Models are not saved by this screening launcher; summaries and metrics are.
This is not exact total-FLOP matching: module routing, final readout, and
parameter counts differ. Use the logged analytical FLOP fields when normalizing
the results.

For fair comparisons, keep sequence length, optimizer steps, effective tokens
per update, data, and seed fixed. Increasing module count changes parameters
and compute. Increasing recurrent depth with tied weights changes compute with
little change to branch parameters. Untied recurrence grows branch parameters
approximately with depth.

## Useful architecture controls

```text
--input-injection initial_only     Embed only at the initial step
--no-share-iteration-weights      Separate branch weights per step
--use-step-embeddings             Add learned recurrent-step identities
--branch-output residual          Branch produces an internal residual update
--exclude-self                    Router cannot select its own prior state
--router-type uniform             Uniform-routing control
--router-type static              Learned input-independent routing control
--router-type identity            No cross-module routing control
--router-heads 4                  Multi-head routing over the module axis
```

Run `python recurrent_attnres_mesh.py --help` for the complete CLI.

## Output files

| Path | Contents |
|---|---|
| `run_args.json` | Fully resolved arguments |
| `model_config.json` | Hugging Face model configuration |
| `hardware.json` | GPU, HBM, BF16, and Flash preflight |
| `metrics.jsonl` | Loss, evaluation, throughput, FLOPs, MFU, and memory |
| `summary.json` | Final comparable run summary |
| `checkpoint-*` | Optional Accelerate model/optimizer/RNG/data state |
| `model/` | Saved HF model, tokenizer, config, and custom source |
| `failure.json` | Structured CUDA OOM details and smaller-batch suggestion |

Each sweep produces a timestamped directory with `sweep_results.csv`, one
directory and `sweep_run.json` per configuration, and `sweep_dry_run.json` for
previews. CUDA OOMs are recorded and the grid continues; other child failures
stop the sweep. Add `--sweep-stop-on-oom` to stop on the first OOM.

## HBM tuning

The baseline is a conservative starting point, not a guarantee that every
possible untied, long-context, or high-module-count ablation fits.

Preserve the 32,768-token effective batch while reducing activation memory:

```bash
--micro-batch-size 4 --gradient-accumulation-steps 4
```

If necessary:

```bash
--micro-batch-size 2 --gradient-accumulation-steps 8
```

Recommended adjustment order:

1. Halve microbatch and double accumulation.
2. Benchmark the exact changed configuration.
3. Keep recurrent checkpointing enabled.
4. Screen with tied weights before large untied experiments.
5. Reduce sequence length if the scientific comparison permits it.
6. Reduce recurrent steps or module count.
7. Use `--no-compile` only to diagnose compilation-specific failures.

Do not disable Flash-SDPA merely to handle an OOM; a fallback attention backend
can require more memory. For a promising configuration with ample measured
headroom, try `--compile-mode max-autotune-no-cudagraphs`. Benchmark before and
after. Disabling checkpointing may improve speed but must also be capacity
tested because it retains more recurrent activations.

## Checkpoints and exact resume

Create periodic state checkpoints:

```bash
OUTPUT_DIR="$PWD/runs/checkpointed" \
  ./scripts/run_train_a100.sh \
  --save-interval 100
```

Resume into a fresh directory:

```bash
OUTPUT_DIR="$PWD/runs/resumed" \
  ./scripts/run_train_a100.sh \
  --resume-state "$PWD/runs/checkpointed/checkpoint-00000500"
```

Exact resume verifies model, tokenizer/data identity, batch and worker settings,
optimizer, schedule, precision, compile/checkpoint policy, seed, process count,
and the stored data cursor. Keep `--max-steps` identical to the original run.

## Hugging Face loading

Saved models include the custom Python source and `auto_map` metadata:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "runs/tinystories/model"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
)
```

The custom architecture disables KV caching; generation recomputes the full
prefix.

## Validation status

The packaged source was validated with syntax/lint checks, the complete smoke
suite, eager and compiled accumulated benchmarks, an Accelerate train/evaluate
probe, profile/CLI/run-config precedence, exact analytical parameter counts,
hardware-preflight simulations, allocator merging, typed/wrapped OOM handling,
and sweep dry runs.

Actual BF16 Flash-SDPA execution, fused AdamW, CUDA compilation, peak HBM fit,
throughput, and MFU still require the first run on a physical A100 80 GB. Start
with `run_benchmark_a100.sh` before launching the full grid.
