# CLAUDE.md

## Project Overview

**ReaLLMASIC nanoGPT** — a research-grade framework extending Andrej Karpathy's nanoGPT for hardware-aware LLM exploration. It provides extensive model architecture variations, training paradigms, quantization, and analysis tools for studying design trade-offs.

## Repository Structure

### Core Files (read these first)

| File | Purpose |
|------|---------|
| `model.py` | GPT model architecture — imports variation modules, supports flash attention, gradient checkpointing, quantization |
| `gpt_conf.py` | `GPTConfig` dataclass with 200+ fields. All model configuration lives here. Supports JSON import/export and per-layer overrides via `*_layerlist` patterns |
| `train.py` | Central training loop with DDP, TensorBoard, checkpoint management, validation, GNS monitoring |
| `train_args.py` | Argparse definitions for all training and model arguments |
| `sample.py` | Inference/sampling with multiple strategies (temperature, top-k, top-p), colorization, benchmarking |

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `variations/` | Model architecture variations (attention, MLP, softmax, norm, activation, position encoding, etc.) |
| `train_variations/` | Training variations — optimizers (30+), loss functions (20+), distillation losses, ETA estimators |
| `data/` | 93 dataset directories. Each has `get_dataset.sh` and symlinks to `data/template/prepare.py` |
| `data/template/` | Master data preparation: `prepare.py`, `tokenizers.py` (11+ tokenizers), `tests.py` |
| `explorations/` | YAML/JSON grid search configs. Use `default_inf.yaml` as template |
| `optimization_and_search/` | Experiment runners: `run_experiments.py`, `run_from_yaml.py`, NSGA genetic search, Vizier |
| `tests/` | Bash and Python test scripts (CPU-based) |
| `analysis/` | Analysis tools: checkpoint, activation, tokenizer, quantization, compression studies |
| `benchmarks/` | BLEU, HellaSwag, softmax sweeps, custom model evaluation |
| `quantization/` | Quantization logic: symmetric, affine, stochastic, ternary methods |
| `utils/` | GPU monitoring, model info/stats, statistical plots, bit usage |
| `documentation/` | Guides for adding variations, contributing features, model stats |
| `huggingface_model/` | HuggingFace model integration scripts |
| `hp_searches/` | Hyperparameter search configs and test scripts |
| `demos/` | End-to-end bash demo scripts for features |
| `initializations/` | Custom weight initialization schemes |
| `distillation/` | Knowledge distillation utilities |

### Supporting Files

| File | Purpose |
|------|---------|
| `shared_param_utils.py` | Parameter sharing across layers (block reuse, symmetry mirroring) |
| `model_merge.py` | Merge two model checkpoints with L2-normalization |
| `hyperparam_search.py` | Hyperparameter search engine |
| `run_exploration_monitor.py` | Textual TUI for monitoring exploration runs |
| `view_hp_log.py` | Textual TUI for hyperparameter search logs |
| `plot_view.py` | Matplotlib/Plotly visualization |
| `view_model_stats.py` | Model statistics comparison |
| `colorize_dataset.py` | Dataset colorization utilities |

## Development Workflow

### Environment Setup

```bash
# CPU
python -m venv venv && source venv/bin/activate
pip install -r requirements_cpu.txt

# GPU
pip install -r requirements_gpu.txt
```

### Typical Workflow

```bash
# 1. Prepare data
cd data/shakespeare_char && bash get_dataset.sh && python prepare.py && cd ../..

# 2. Train
python train.py --dataset=shakespeare_char --device=cpu --max_iters=100

# 3. Sample/inference
python sample.py --out_dir=out-shakespeare_char --device=cpu
```

### Training Paradigms

- **Standard**: `train.py` — full backpropagation with DDP support
- **MeZO**: `train_mezo.py` — zeroth-order (forward-only) training via perturbations
- **Recurrent latent**: `train_recurrent.py` — latent-chaining fine-tuning from existing checkpoints

### Running Tests

```bash
# Individual test scripts
bash tests/test_all_activation_variations_cpu.sh
bash tests/test_all_softmax_variations_cpu.sh
bash tests/test_finetuning_cpu.sh
bash tests/test_quantization_cpu.sh
bash tests/test_run_experiments.sh

# Python tests
python tests/test_bit_balanced_loss.py
python tests/test_numerical_multicontext_fp16.py

# Tokenizer tests
python data/template/tests.py
```

All CI tests are CPU-based (see `.github/workflows/`).

## Architecture & Configuration

### Model Variations

The model is highly modular. Key variation categories in `variations/`:

- **Attention**: CausalSelfAttention, LinearAttention, InfiniteHeadAttention, MultiHeadLatentAttention, Co4Attention, EdgeLLMASIC, Flash Lobo
- **MLP**: OriginalMLP, SwiGLU, DualPathMLP, KanMLP, MoE, EdgeLLMASIC
- **Softmax**: softmax, softermax, sigsoftmax, consmax, strongermax, polymax, exppolymax, relu*max, sigmoid*max
- **Normalization**: LayerNorm, RMSNorm, DACT, HSNorm, KRMSNorm
- **Position encoding**: Absolute, Rotary (RoPE), FIRE
- **Activation**: GELU, ReLU, SiLU, Tanh, Shifted GELU, learnable activations

### Configuration System

1. **`GPTConfig`** (`gpt_conf.py`): Dataclass with all model parameters
2. **Per-layer overrides**: Use `*_layerlist` parameters (e.g., `n_head_layerlist`, `attention_variant_layerlist`)
3. **CLI args**: `train_args.py` maps argparse → GPTConfig
4. **JSON**: `GPTConfig.from_json()` / `GPTConfig.to_json()`
5. **Explorations**: YAML files in `explorations/` for grid searches

### Adding New Variations

When adding a new model variation:
1. Add implementation to the appropriate file in `variations/`
2. Add configuration fields to `gpt_conf.py`
3. Add CLI arguments to `train_args.py`
4. Wire it up in `model.py`
5. Add tests
6. See `documentation/Adding_Variations.md` for the full guide

## Conventions & Rules

### Code Style

- Keep changes minimal and task-focused
- Match the style and structure of surrounding code
- No unrelated refactors alongside feature work
- Do not commit binaries or image files

### When Modifying Variations

- Changes to `variations/` may require updates to `gpt_conf.py` and `train_args.py`
- Changes to `run_experiments.py` may require updates to `run_exploration_monitor.py`
- Changes to `hyperparam_search.py` may require updates to `view_hp_log.py` and new test scripts in `hp_searches/`
- Tokenizer changes go in `data/template/tokenizers.py` with tests in `data/template/tests.py`

### Adding Datasets

Each new dataset directory under `data/` must have:
- Symlinks to `data/template/utils` and `data/template/prepare.py`
- A `get_dataset.sh` script that produces `input.txt`
- A README with dataset description, source link, and license

### Exploration Configs

- Use `common_group` to keep output file names short
- Use `named_static_groups` and `named_group_static` for compact parameter specifications
- Use `named_group-alternate` for varying across groups
- Template: `explorations/default_inf.yaml`

### Commit Messages

Format: `<type(scope): summary>`

Examples:
- `feat(attention): add multi-head latent attention variant`
- `fix(train): correct gradient accumulation with DDP`
- `test(softmax): add consmax_v2 variation tests`

### Validation Before Finishing

- [ ] All relevant tests pass locally
- [ ] No unrelated refactors included
- [ ] No binaries or image files committed
- [ ] New features include tests when applicable

## .gitignore Notes

Excluded: `__pycache__/`, `logs/`, `csv_logs/`, `exploration_logs/`, `out*/`, `*.pkl`, `*.bin`, `*.txt`, `*.wav`, `*.mp3`, `experiments/`, `nsga_exps/`, `venv/*`, `.aider*`

Note: `*.txt` is gitignored — data files and logs are not tracked.

## CI/CD

GitHub Actions workflows in `.github/workflows/` (all CPU-based, triggered on PRs):
- Basic install → prepare → train → inference pipeline
- Feature-specific tests: activations, softmax, finetuning, GQA, gradient checkpointing, quantization, parallel MLP, tokenization, experiment runner
