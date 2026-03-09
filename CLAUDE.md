# CLAUDE.md — AI Assistant Guide for ReaLLMASIC/nanoGPT

This document describes the codebase structure, development workflows, and key
conventions for AI assistants working in this repository.

---

## Project Overview

**ReaLLMASIC** (formerly nanoGPT) is a hardware-aware LLM research framework
built on top of Andrej Karpathy's nanoGPT. Its primary goals are:

- Exploring model variations (attention, MLP, softmax, normalization, etc.)
- Evaluating power-performance-area (PPA) implications of design choices
- Supporting diverse tokenization schemes and datasets
- Enabling reproducible hyperparameter searches and explorations

---

## Repository Layout

```
nanoGPT/
├── model.py                  # Core GPT model definition
├── gpt_conf.py               # GPTConfig dataclass (all model hyperparameters)
├── train.py                  # Main training script (Trainer class)
├── train_args.py             # Argparse argument definitions for train.py
├── train_mezo.py             # Forward-only (MeZO) optimizer training
├── train_recurrent.py        # Recurrent training variant
├── sample.py                 # Inference / generation script
├── hyperparam_search.py      # Hyperparameter search utilities
├── view_hp_log.py            # HP log viewer
├── view_model_stats.py       # Per-tensor stats table viewer
├── model_merge.py            # Model merging utilities
├── shared_param_utils.py     # Shared-parameter utilities
│
├── variations/               # Pluggable module variations (see below)
├── train_variations/         # Training-side variations (optimizers, loss, etc.)
├── quantization/             # Quantization utilities
├── data/                     # Dataset preparation scripts (one dir per dataset)
├── explorations/             # YAML/JSON sweep configs for run_experiments.py
├── optimization_and_search/  # run_experiments.py and NSGA search
├── tests/                    # Bash CI test scripts
├── .github/workflows/        # GitHub Actions CI definitions
├── demos/                    # Runnable demo scripts
├── documentation/            # Lightweight feature docs
├── analysis/                 # Post-training analysis tools
├── logging/                  # Tensorboard / CSV log helpers
├── benchmarks/               # Benchmark suite (called from sample.py)
├── utils/                    # GPU monitoring, model info, stats
├── hp_searches/              # Hyperparameter search configs
├── initializations/          # Weight initialization experiments
├── distillation/             # Knowledge distillation tools
├── quantizations/            # Additional quantization experiments
├── huggingface_model/        # HuggingFace model integrations (Gemma, PaLI-Gemma)
├── exutorch/                 # ExecuTorch mobile export
└── report/                   # Reporting utilities
```

---

## Core Files

### `model.py`
Full GPT model definition. Imports variation dictionaries from `variations/`
and instantiates modules based on `GPTConfig` settings.

Key classes:
- `GPT` — top-level model with `forward()`, `generate()`, `from_pretrained()`,
  `configure_optimizers()`, and `estimate_mfu()`
- `Block` — a single transformer block (attention + MLP with configurable norm)
- `CausalSelfAttention` — default attention, switched via `attention_variant`

### `gpt_conf.py`
Single source of truth for **all** model hyperparameters. `GPTConfig` is a
`@dataclass` with defaults. Key groups of fields:

| Group | Key Fields |
|---|---|
| Architecture | `n_layer`, `n_head`, `n_embd`, `block_size`, `vocab_size` |
| Attention | `attention_variant`, `n_kv_group`, `use_flash_lobo`, `use_qk_norm` |
| MLP | `mlp_variant`, `mlp_expansion_factor`, `mlp_size`, `use_parallel_mlp` |
| Softmax | `softmax_variant_attn`, `softmax_variant_output` |
| Normalization | `norm_variant_attn`, `norm_variant_output`, `use_pre_ln`, `use_post_ln`, `use_peri_ln` |
| Positional Encoding | `use_abs_pos_embeddings`, `use_rotary_embeddings`, `rope_variant`, `use_fire_embeddings` |
| MoE | `use_moe`, `n_experts`, `moe_top_k`, `moe_router_scheme` |
| Quantization | `quantize_wte`, `quantize_linear_method`, `quantize_linear_bits`, etc. |
| Weight Tying | `wte_weight_tying`, `n_embd_wte` (factorized embedding) |

`GPTConfig.from_json(filename)` / `.to_json(filename)` allow checkpoint-safe
serialization.

### `train.py`
Contains the `Trainer` class. Entry point is `python3 train.py [args]`.

Key `Trainer` responsibilities:
- Loading/resuming checkpoints (`--init_from resume|scratch|gpt2*`)
- Distributed Data Parallel (DDP) support via `torchrun`
- Gradient accumulation, AMP, and `torch.compile`
- Evaluation loop with optional sampling (`--max_sample_tokens`)
- Tensorboard and CSV logging
- GNS (Gradient Noise Scale) monitoring via hooks
- Exporting WTE embeddings and scale matrices as `.npy`/`.npz`

### `train_args.py`
All `argparse` arguments for `train.py`. Arguments are sorted into three groups:

| Group | Purpose |
|---|---|
| `model_group` | Auto-forwarded into `GPTConfig` |
| `training_group` | Training loop settings (lr, batch, iters, etc.) |
| `logging_group` | Logging/monitoring settings |

**Important**: Adding a new model feature requires entries in **both**
`gpt_conf.py` (field) and `train_args.py` (argparse argument in `model_group`).

### `sample.py`
Inference script. Key function `sample_with_existing_model()` can also be
called from within `train.py` during evaluation.

---

## Variations System

All swappable modules live in `variations/` and expose **dictionaries** mapping
string keys to classes/functions. `model.py` selects from these dictionaries
using the corresponding `GPTConfig` field.

| File | Config Key(s) | Examples |
|---|---|---|
| `activation_variations.py` | `activation_variant` | `gelu`, `relu`, `silu`, `squared_relu`, `pla`, `pfla` |
| `attention_variations.py` | `attention_variant` | `causal`, `mla`, `co4`, `ssm`, `flex` |
| `mlp_variations.py` | `mlp_variant` | `mlp`, `swiglu`, `kan` |
| `norm_variations.py` | `norm_variant_attn/output` | `rmsnorm`, `layernorm`, `hyperspherenorm`, `krmsnorm` |
| `softmax_variations.py` | `softmax_variant_attn/output` | `softmax`, `consmax`, `strongermax`, `polymax`, `softermax` |
| `position_encoding_variations.py` | `use_rotary_embeddings`, `use_fire_embeddings` | `RotaryEmbedding`, `FIRE`, `SymmetricalOverlapAngularPositions` |
| `linear_variations.py` | `linear_variant_attn/mlp` | `linear`, `bitlinear`, `adaptive_linear` |
| `moe_variations.py` | `use_moe` | `MoELayer` |
| `lsv_variations.py` | `use_lsv` | Learned Steering Vectors |
| `router_variations.py` | `moe_router_scheme` | `softmax`, various |
| `model_variations.py` | `--model_variation` CLI flag | Entire pre-configured model archs |

### Adding a New Variation (5-step process)

1. **Add class** to the relevant `variations/*.py` file and register it in the
   module's dictionary.
2. **Import + wire** in `model.py` — use the dictionary lookup pattern:
   `self.act = activation_dictionary[config.activation_variant](config)`.
3. **Add field** to `GPTConfig` in `gpt_conf.py` with a sensible default.
4. **Add argparse argument** in `train_args.py` under `model_group`.
5. **Create exploration YAML** in `explorations/` covering the new option.

See `documentation/Adding_Variations.md` for the canonical guide.

---

## Training Variations

`train_variations/` holds training-side pluggable components:

| File | Contents |
|---|---|
| `optimizer_variants.py` | `optimizer_dictionary`: `adamw`, `muon`, `adabelief`, `adan`, `sgd`, `act_reg_adamw`, etc. |
| `loss_variants.py` | `LOSS_VARIANTS`: `cross_entropy`, `bit_balanced`, etc. |
| `distillation_loss_variants.py` | `DISTILLATION_LOSS_VARIANTS` |
| `eta_variants.py` | ETA/latency estimators |
| `muon.py` | Muon optimizer implementation |

---

## Dataset System

Each dataset lives in `data/<name>/` with:
- `get_dataset.sh` or `prepare.py` — downloads + tokenizes into `train.bin` / `val.bin`
- `README.md` — dataset description

Datasets are specified at train time with `--dataset <name>` (strips leading
`data/` automatically).

Multi-dataset training: `--multicontext` flag enables training on multiple
datasets simultaneously.

Common datasets: `shakespeare_char`, `openwebtext`, `minipile`, `fineweb-edu`,
`wikitext103`, `gsm8k`, `mmlu`, `midi_jsbach`, and many more.

---

## Exploration / Hyperparameter Sweeps

`explorations/` contains YAML (preferred) and JSON sweep configurations.

### Running a Sweep

```bash
python3 optimization_and_search/run_experiments.py -c explorations/config.yaml
```

Outputs land in:
- `csv_logs/` — bulk CSV training logs
- `logs/` — text logs
- `out/` (or `--output_dir`) — timestamped checkpoint subdirectories

### YAML Structure

```yaml
named_static_groups:           # atomic reusable param groups
  - named_group: "rotary"
    use_rotary_embeddings: [true]
    use_abs_pos_embeddings: [false]

common_group:                  # applied to every run (omitted from run name)
  dataset: ["minipile"]
  max_iters: [10000]

parameter_groups:              # cartesian product of variations to sweep
  - compile: [true]
    named_group_static: ["rotary"]
```

### Monitoring

```bash
# Real-time best val loss dashboard
watch --color 'python3 checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss'

# Live CSV chart (ASCII, refreshes every 5s)
python3 logging/view_csv_logs.py --csv-dir csv_logs --pattern "**/bulk_*.csv"

# Tensorboard
source ./logging/start_tensorboard.sh
```

---

## Inference

```bash
# Basic inference from last checkpoint
python3 sample.py

# With options
python3 sample.py --out_dir out --max_new_tokens 200 --temperature 0.8 --top_k 200
```

Key `sample.py` arguments:
- `--init_from resume|gpt2|gpt2-medium|gpt2-large|gpt2-xl`
- `--start "Your prompt"` or `--start FILE:prompt.txt`
- `--num_samples`, `--max_new_tokens`, `--temperature`, `--top_k`
- `--compile` — enable `torch.compile` for faster inference

---

## Quickstart Workflow

```bash
# 1. Prepare a dataset
bash data/shakespeare_char/get_dataset.sh

# 2. Train
python3 train.py --compile --max_sample_tokens 100

# 3. Inference
python3 sample.py

# 4. Run a sweep
python3 optimization_and_search/run_experiments.py -c explorations/sample.yaml

# 5. Monitor
watch --color 'python3 checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss'
```

### MeZO (forward-only gradient estimation)

```bash
python3 train_mezo.py --dataset shakespeare_char --max_iters 2000
```

---

## Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `shakespeare_char` | Dataset name (from `data/`) |
| `--out_dir` | `out` | Checkpoint output directory |
| `--max_iters` | 5000 | Training iterations |
| `--batch_size` | 12 | Micro-batch size |
| `--block_size` | 1024 | Context length |
| `--n_layer` | 12 | Number of transformer layers |
| `--n_head` | 12 | Attention heads |
| `--n_embd` | 768 | Embedding dimension |
| `--learning_rate` | 6e-4 | Peak learning rate |
| `--compile` | False | Enable `torch.compile` |
| `--init_from` | `scratch` | `scratch`, `resume`, or `gpt2*` |
| `--eval_interval` | 250 | Evaluate every N iters |
| `--max_sample_tokens` | None | Generate N tokens at each eval |
| `--optimizer` | `adamw` | Optimizer variant |
| `--device` | `cuda` | Training device |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, `float32` |
| `--wandb_log` | False | Enable Weights & Biases logging |

---

## Testing

### Running Tests

All tests are CPU-based bash scripts (no GPU required):

```bash
bash tests/test_all_softmax_variations_cpu.sh
bash tests/test_all_activation_variations_cpu.sh
bash tests/test_quantization_cpu.sh
bash tests/test_finetuning_cpu.sh
bash tests/test_run_experiments.sh
# etc.
```

CI tests run automatically on every PR and commit via `.github/workflows/`.

### Adding a Test

1. Create `tests/test_<feature>_cpu.sh` — use a tiny network (CPU-sized).
2. Wire it into a `.github/workflows/*.yml` file.
3. Tests run in parallel; keep them small and self-contained.

---

## Model Statistics and Analysis

```bash
# Generate per-tensor stats CSV during training
python3 train.py --print_model_stats_table run1_stats.csv

# Compare two runs
python3 view_model_stats.py run1_stats.csv run2_stats.csv

# Inspect checkpoints
python3 checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss
```

---

## Distributed Training

```bash
torchrun --standalone --nproc_per_node=<N_GPUS> train.py --compile
```

DDP is handled automatically when `torchrun` sets `RANK` / `LOCAL_RANK`
environment variables.

---

## Dependencies

### GPU Setup
```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install -r requirements_gpu.txt
```

### CPU-only
```bash
pip install -r requirements_cpu.txt
```

Key packages: `torch`, `numpy`, `transformers`, `datasets`, `tiktoken`,
`sentencepiece`, `wandb`, `tensorboard`, `rich`, `torchinfo`, `seaborn`.

---

## Conventions for AI Assistants

### Code Style
- Python 3.10+ with type hints where present (use `|` union syntax, not `Optional`)
- No docstrings or extra comments on code you didn't change
- Keep changes minimal — only modify what is explicitly required

### Adding Features
- Follow the 5-step variation pattern (see [Adding a New Variation](#adding-a-new-variation-5-step-process))
- Always add the `GPTConfig` field **and** the `train_args.py` argparse entry together
- Register new variations in the module's dictionary — never use raw `if/elif` chains outside `model.py`
- The `model_group` argparse arguments are **automatically forwarded** to `GPTConfig`; `training_group` and `logging_group` are not

### Config Precedence
`GPTConfig` defaults → command-line args → `--init_from resume` (loads saved config)

### Checkpoint Format
Checkpoints (`ckpt.pt`) saved by `train.py` contain:
```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': dict,   # GPTConfig fields
    'iter_num': int,
    'best_val_loss': float,
    'config': args_namespace,
}
```

### Exploration YAML vs JSON
- **YAML** (`.yaml`) is the preferred modern format — supports named groups
- **JSON** (`.json`) is legacy but still supported
- Use `explorations/sample.yaml` as the canonical template

### Documentation
- New variation → `documentation/<FeatureName>.md` (lightweight, link any paper)
- New demo → `demos/<feature>_demo.sh`
- Keep `README.md` minimal; deeper docs go in `documentation/`

### PR Workflow
- Post PR early for feedback; merge as soon as the feature works
- Every new feature should have at least one exploration config and one test script
- Tests must run on CPU and finish quickly (CI has no GPU)
