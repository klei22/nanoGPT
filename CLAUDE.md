# CLAUDE.md

This file provides context and guidelines for AI assistants working in the
**ReaLLMASIC / nanoGPT** codebase — a hardware-aware language model research
framework that extends Karpathy's nanoGPT with extensive modular variations,
exploration tooling, and hardware/PPA analysis.

---

## Repository Purpose

ReaLLMASIC bridges theoretical model design and practical hardware
implementation. Key goals:

- Plug-and-play model component variations (attention, MLP, norm, softmax, etc.)
- Hardware-aware training (quantization, PPA analysis, GPU memory monitoring)
- Systematic exploration of hyperparameter and architecture spaces
- Support for diverse datasets, tokenizers, and training paradigms

---

## Core File Map

| File | Role |
|------|------|
| `train.py` | Central training orchestrator (`Trainer` class, ~2200 lines) |
| `model.py` | GPT model definition — imports from `variations/` |
| `gpt_conf.py` | `GPTConfig` dataclass: 100+ model configuration options |
| `train_args.py` | All CLI arguments for `train.py` and other training scripts (200+ args) |
| `sample.py` | Inference/sampling engine with visualization |
| `train_mezo.py` | MeZO (forward-only, zeroth-order) training |
| `train_recurrent.py` | Recurrent architecture training |
| `shared_param_utils.py` | Shared parameter group utilities |
| `hyperparam_search.py` | Hyperparameter sweep runner |
| `run_exploration_monitor.py` | TUI monitor for `run_experiments.py` output (update if `run_experiments.py` changes) |
| `view_hp_log.py` | Monitor and explore `hyperparam_search.py` logs |
| `model_merge.py` | Utilities for merging model checkpoints |

---

## Directory Structure

```
variations/          Model architecture component variations
train_variations/    Training methodology variations (optimizers, losses)
data/                Datasets — 93+ subdirectories, each with prepare.py
data/template/       Shared tokenizer definitions and prepare pipeline
explorations/        YAML/JSON grid-search configs
optimization_and_search/  Search orchestration scripts
utils/               GPU monitoring, model stats, introspection
analysis/            Post-hoc analysis scripts (17+ subdirectories)
huggingface_model/   HuggingFace model integration (Gemma, PaliGemma, etc.)
quantization/        Quantization methods and utilities
tests/               Test suite
demos/               End-to-end demo bash scripts
hp_searches/         Test scripts and saved settings for hyperparam_search.py
logging/             Tensorboard/CSV log utilities
report/              Research papers (ICLR, NeurIPS, ICML)
```

---

## Key Architectural Conventions

### Model Variations (`variations/`)

Every model component is pluggable. When adding a new component type:

1. Place the implementation in the appropriate `variations/*.py` file.
2. Update `gpt_conf.py` to add any new config fields.
3. Update `train_args.py` to expose new CLI flags.
4. Wire the variant into the parent module (usually `model.py` or the relevant block).

| File | Component |
|------|-----------|
| `attention_variations.py` | `CausalSelfAttention`, `LinearAttention`, `InfiniteAttention`, `MLA`, `Co4` |
| `mlp_variations.py` | `OriginalMLP` and variants with L2-norm, offsets, quantization |
| `softmax_variations.py` | `Softermax`, `ConSmax`, `SigSoftmax`, and 5+ more |
| `activation_variations.py` | 28+ activation functions including learnable variants |
| `norm_variations.py` | `LayerNorm`, `RMSNorm`, pre/post/peri variants |
| `position_encoding_variations.py` | Rotary, FIRE, Symmetric, dynamic scale |
| `linear_variations.py` | Linear layers with optional quantization |
| `lsv_variations.py` | Low-rank + sparse variants |
| `moe_variations.py` | Mixture of Experts layer |
| `router_variations.py` | MoE routing strategies |
| `block_variations.py` | Transformer block compositions |
| `model_variations.py` | High-level model structure switcher |

### Training Variations (`train_variations/`)

| File | Component |
|------|-----------|
| `optimizer_variants.py` | 20+ optimizers (Adam, SGD, Lion, Muon, AdaBelief, Adan, …) |
| `loss_variants.py` | 15+ loss functions (CE, focal, rank-distance, label-smoothing, bit-balanced) |
| `distillation_loss_variants.py` | Knowledge distillation variants |
| `eta_variants.py` | ETA estimation strategies |
| `muon.py` | Muon optimizer implementation |

---

## GPTConfig (`gpt_conf.py`)

The `GPTConfig` dataclass drives every aspect of the model. Notable parameter
groups:

- **Core shape:** `block_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`
- **MLP:** `mlp_expansion_factor`, `mlp_size`, `mlp_down_projs`
- **Attention:** `attention_list`, `n_kv_group` (grouped-query attention)
- **Normalization:** `use_pre_ln`, `use_post_ln`, `use_peri_ln`; `norm_variant_*`
- **Position encoding:** `use_rotary_embeddings`, `use_abs_pos_embeddings`, `shared_fire_embeddings`
- **Per-layer variability:** `n_head_layerlist`, `mlp_size_layerlist`, `attention_variant_layerlist`, `window_size_layerlist`
- **Quantization:** 20+ quantization parameters
- **Numerical multicontext (time-series):** `numerical_multicontext`, `numerical_embedding_variant`, `numerical_output_variant`

When adding new model features, add corresponding fields here and expose them
via `train_args.py`.

---

## Data Pipeline (`data/`)

### Adding a New Dataset

Each dataset directory must contain:
- `get_dataset.sh` — downloads and creates `input.txt` (or equivalent)
- `prepare.py` (or symlink to `data/template/prepare.py`)
- Symlinks to `data/template/utils` and `data/template/prepare.py`
- A brief `README.md` describing the dataset, source URL, and license

### Tokenizers (`data/template/tokenizers.py`)

11 tokenization methods are available:

| Tokenizer | Use case |
|-----------|----------|
| `TiktokenTokenizer` | GPT-2 / CL100k / R50k encodings |
| `SentencePieceTokenizer` | Language-agnostic BPE |
| `CharTokenizer` | Character-level |
| `ByteTokenizer` | Byte-level |
| `CustomTokenizer` | User-defined token lists |
| `CharBPE` | Character + byte fallback |
| `JsonByteTokenizer` | JSON-based + byte fallback |
| `PythonProgrammingTokenizer` | Code-focused tokens |
| `SineWaveTokenizer` | Synthetic sine wave data |
| `WhisperMelCsvTokenizer` | Audio mel-spectrogram |

When adding a new tokenizer, also add tests in `data/template/tests.py`.

---

## Training Workflows

### Standard Training

```bash
# Prepare data
bash data/shakespeare_char/get_dataset.sh

# Train (GPU recommended)
python3 train.py --compile

# Train with periodic sampling output
python3 train.py --max_sample_tokens 100 --compile
```

### MeZO (Forward-Only) Training

```bash
# From scratch
python3 train_mezo.py --dataset shakespeare_char --max_iters 2000 \
    --batch_size 64 --block_size 256

# Resume from checkpoint
python3 train_mezo.py --init_from resume --out_dir out

# Adjust perturbation scale
python3 train_mezo.py --mezo_epsilon 1e-3 --mezo_seed 42
```

### Inference / Sampling

```bash
python3 sample.py
python3 sample.py --out_dir out --colorize_output --show_heatmaps
```

### Exploration / Grid Search

```bash
# Run a grid search defined in a YAML config
python3 optimization_and_search/run_experiments.py -c explorations/config.yaml

# Monitor progress in TUI
python3 run_exploration_monitor.py

# Inspect checkpoints by validation loss
python3 analysis/checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss

# Live watch variant
watch --color 'python3 analysis/checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss'
```

### Hyperparameter Search

```bash
python3 hyperparam_search.py --config hp_searches/config.yaml
python3 view_hp_log.py --directory hp_logs
```

When modifying `hyperparam_search.py` or `view_hp_log.py`, add a new test `.sh`
script and YAML file in `hp_searches/`.

### Logging

```bash
# Start TensorBoard
source ./logging/start_tensorboard.sh           # port 6006
source ./logging/start_tensorboard.sh 6007      # alternate port

# View CSV logs (tensorboard-style ASCII)
python3 logging/view_csv_logs.py --csv-dir csv_logs --pattern "**/bulk_*.csv"
```

### Model Stats

```bash
# Export per-tensor statistics to CSV during training
python3 train.py --print_model_stats_table stats.csv
```

---

## Exploration YAML Conventions (`explorations/`)

Use `explorations/default_inf.yaml` as a reference template.

**Best practices:**
- Use `common_group` for parameters shared across all runs (keeps output
  filenames short).
- Use `named_static_groups` to define reusable parameter bundles.
- Reference them via `named_group_static` in individual run specs.
- Use `named_group_alternate` to sweep across named groups.

---

## Visualization Coupling

`run_exploration_monitor.py` parses YAML files produced by
`optimization_and_search/run_experiments.py`. **If `run_experiments.py` is
updated, check whether `run_exploration_monitor.py` also needs updating.**

---

## Tests

```
tests/test_numerical_multicontext_fp16.py   FP16 numerical multicontext validation
tests/test_bit_balanced_loss.py             Bit-balanced loss correctness
data/template/tests.py                      Tokenizer validation suite
hp_searches/                                Hyperparameter search test scripts
```

Run tests before finishing any PR. When fixing bugs or adding features, add or
update the relevant tests.

---

## Dependencies

Install for GPU (CUDA 11.8+):

```bash
python3 -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install numpy transformers datasets tiktoken wandb tqdm \
    tensorboard rich torchinfo
```

Key packages: `torch`, `numpy`, `transformers` (4.44.2), `datasets` (2.21.0),
`tiktoken` (0.7.0), `wandb` (0.18.3), `sentencepiece`, `plotly` (5.22.0),
`rich` (14.0.0), `torchinfo` (1.8.0).

Full CPU and GPU requirement files: `requirements_cpu.txt`, `requirements_gpu.txt`.

---

## Commit / PR Conventions

- **Commit title format:** `<type(scope): summary>`
  - Example: `feat(attention): add MLA lobo scaling`
- Keep changes minimal and task-focused.
- Do not commit binaries, images, or unrelated refactors.
- Do not commit `.vscode/` settings or generated `meta.pkl` / `.bin` data files.

---

## Validation Gate (before finishing any task)

- [ ] All required commands pass locally.
- [ ] New tokenizers have tests in `data/template/tests.py`.
- [ ] New datasets have `get_dataset.sh`, `README.md`, and correct symlinks.
- [ ] New hyperparameter search changes have a test in `hp_searches/`.
- [ ] No binaries or image files committed.
- [ ] No unrelated refactors.
