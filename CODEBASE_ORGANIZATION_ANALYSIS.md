# nanoGPT Codebase Organization Analysis

## Current Structure Overview

The repo has grown significantly beyond the original "all in one file" nanoGPT philosophy. It now contains **200+ Python files** across **100+ directories**, spanning model architecture, training, data preparation, analysis, quantization, hardware design, and more.

### Current Top-Level Layout

```
nanoGPT/
├── model.py              (981 lines)  - GPT model definition
├── train.py              (2230 lines) - Main training loop
├── sample.py             (1593 lines) - Inference/sampling
├── gpt_conf.py           (578 lines)  - GPTConfig dataclass
├── train_args.py         (1487 lines) - Argparse definitions
├── shared_param_utils.py (285 lines)  - Parameter sharing utilities
├── variations/           (16 files)   - Model architecture variants
├── train_variations/     (5 files)    - Training algorithm variants
├── quantization/         (6 files)    - Quantization tools (including ptq/)
├── initializations/      (4 files)    - Weight initialization variants
├── utils/                (6 files)    - GPU monitoring, model stats, plots
├── data/                 (80+ dirs)   - Dataset preparation scripts
├── analysis/             (15+ dirs)   - Research analysis scripts
├── benchmarks/           (4 files)    - Benchmarking tools
├── explorations/                      - Experiment configs (JSON/YAML)
├── hp_searches/                       - Hyperparameter search configs
├── optimization_and_search/           - NSGA/Vizier search infrastructure
├── HW/                                - Verilog hardware designs
├── huggingface_model/                 - HF model integration
├── distillation/                      - Distillation experiments
├── report/                            - Paper/report LaTeX
├── documentation/                     - Internal docs
├── demos/                             - Demo scripts
├── colabs/                            - Colab notebooks
├── logging/                           - Log viewing tools
├── exutorch/                          - ExecuTorch export
├── tf_np_golden_gen/                  - TF/NumPy golden generation
├── util_factorization/                - Embedding factorization tools
├── tests/                             - Tests (only 3 files)
├── dev_env_setup_scripts/             - Environment setup
├── 10+ standalone .py scripts         - Various utilities at root level
└── 2 shell scripts                    - Exploration/report helpers
```

---

## Key Issues

### 1. Root Directory Clutter (High Priority)

**14 Python files + 2 shell scripts** sit at the root alongside the core files. Many are utilities or viewers that don't need top-level prominence:

- `analyze_numerical_output.py` - analysis tool
- `colorize_dataset.py` - data visualization tool
- `plot_view.py` - plot viewer
- `view_hp_log.py` - hyperparameter log viewer
- `view_model_stats.py` - model statistics viewer
- `hyperparam_search.py` - HP search runner
- `run_exploration_monitor.py` - exploration monitor
- `model_merge.py` - model merging utility
- `compile_report.sh` - report compilation
- `create_new_exploration.sh` - exploration scaffolding

**Suggestion:** Move standalone tool scripts into `tools/` or `scripts/`, keeping only the core trinity (`model.py`, `train.py`, `sample.py`) plus config files at the root.

### 2. Oversized Files (High Priority)

| File | Lines | Concern |
|------|-------|---------|
| `train.py` | 2230 | Mixes training loop, logging, checkpointing, data loading, evaluation, and rich UI |
| `sample.py` | 1593 | Handles sampling, tokenizer loading, interactive mode, benchmark eval — too many responsibilities |
| `train_args.py` | 1487 | Monolithic argparse with 300+ arguments, no grouping by category |
| `train_variations/optimizer_variants.py` | 1634 | Many custom optimizer implementations in one file |
| `variations/attention_variations.py` | 1546 | Many attention implementations in one file |
| `gpt_conf.py` | 578 | Config dataclass with 100+ fields, no sub-configs |

**Suggestions:**
- **`train.py`**: Extract data loading, checkpointing/saving, evaluation, and the rich UI into separate modules under a `training/` package.
- **`sample.py`**: Split tokenizer management, interactive REPL, and batch sampling into separate modules.
- **`train_args.py`**: Group arguments into sub-parsers or separate files by category (model args, training args, logging args, data args).
- **`gpt_conf.py`**: Consider nested config dataclasses (e.g., `AttentionConfig`, `MLPConfig`, `NumericalConfig`) to reduce the flat field sprawl.

### 3. Duplicate and Overlapping Directories (Medium Priority)

- ~~**`quantization/` vs `quantizations/`**~~ — **DONE**: Merged `quantizations/ptq/` into `quantization/ptq/`.
- **`util_factorization/`** — Could live under `utils/` or `tools/`.
- **`logging/`** has only one file (`view_csv_logs.py`) — could merge into `utils/`.
- **`tests/`** has only 3 files with no `__init__.py` — very low test coverage for a project this size.

### 4. `data/` Directory Scale (Medium Priority)

80+ dataset directories, many with their own `prepare.py`, `get_dataset.py`, and utility scripts. The `data/template/` directory provides shared tooling but the pattern is inconsistent:

- Some datasets use `get_dataset.py`, others use `prepare.py`, some have both.
- Utility functions are duplicated across dataset dirs (e.g., `partition_file.py` exists in both `data/template/utils/` and `data/wikipedia/`).
- `data/template/utils/` has 25+ utility scripts that could be a proper importable package.

**Suggestions:**
- Standardize the dataset preparation interface: every dataset should have one entry point (e.g., `prepare.py`) that follows a common pattern.
- Make `data/template/utils/` a proper Python package and import from it rather than duplicating scripts.
- Consider a `datasets/` rename to avoid confusion with the `data/` concept (raw data files vs. preparation code).

### 5. Missing `__init__.py` Files (Medium Priority)

Many directories that are used as Python packages lack `__init__.py` files, relying on implicit namespace packages. This makes it harder to understand the public API of each module. Key directories that should have `__init__.py`:

- `variations/`
- `train_variations/`
- `utils/`
- `quantization/`
- `benchmarks/` (has one already)

### 6. Scattered Experiment/Research Code (Low Priority)

Research artifacts are spread across many top-level directories:
- `analysis/` — mathematical analysis scripts
- `distillation/` — distillation experiments
- `HW/` — hardware (Verilog) design
- `huggingface_model/` — HF integration experiments
- `initializations/` — initialization research
- `tf_np_golden_gen/` — TF golden generation
- `report/` — paper drafts

**Suggestion:** Group these under a `research/` or `experiments/` umbrella directory to reduce top-level noise. The core training infrastructure should be immediately visible.

---

## Proposed Reorganized Structure

```
nanoGPT/
├── model.py                          # Core model (keep at root)
├── train.py                          # Main training entry point (keep at root)
├── sample.py                         # Sampling entry point (keep at root)
├── gpt_conf.py                       # Model config (keep at root)
├── train_args.py                     # Training args (keep at root)
│
├── model/                            # Model architecture components
│   ├── __init__.py
│   ├── variations/                   # (moved from variations/)
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── mlp.py
│   │   ├── softmax.py
│   │   ├── ...
│   ├── quantization/                 # (merged quantization/ + quantizations/)
│   │   ├── __init__.py
│   │   ├── quantize.py
│   │   ├── ptq/
│   │   └── ...
│   └── initializations/              # (moved from initializations/)
│
├── training/                         # Training infrastructure
│   ├── __init__.py
│   ├── optimizer_variants.py         # (from train_variations/)
│   ├── loss_variants.py
│   ├── distillation.py
│   ├── data_loading.py              # (extracted from train.py)
│   ├── checkpointing.py            # (extracted from train.py)
│   └── evaluation.py               # (extracted from train.py)
│
├── utils/                            # Shared utilities (expanded)
│   ├── __init__.py
│   ├── gpu_monitoring.py
│   ├── model_info.py
│   ├── model_stats.py
│   ├── statistic_plots.py
│   ├── shared_params.py             # (from shared_param_utils.py)
│   └── bit_usage.py
│
├── tools/                            # Standalone CLI tools
│   ├── analyze_numerical_output.py
│   ├── colorize_dataset.py
│   ├── hyperparam_search.py
│   ├── model_merge.py
│   ├── plot_view.py
│   ├── view_hp_log.py
│   ├── view_model_stats.py
│   └── run_exploration_monitor.py
│
├── data/                             # Dataset preparation
│   ├── common/                       # (from data/template/utils/, as package)
│   │   ├── __init__.py
│   │   ├── tokenizers.py
│   │   ├── prepare.py
│   │   └── ...
│   ├── shakespeare_char/
│   ├── openwebtext/
│   └── ...
│
├── configs/                          # Experiment configurations
│   ├── explorations/                 # (from explorations/)
│   └── hp_searches/                  # (from hp_searches/)
│
├── research/                         # Research & analysis code
│   ├── analysis/
│   ├── distillation/
│   ├── hardware/                     # (from HW/)
│   ├── huggingface/                  # (from huggingface_model/)
│   ├── reports/                      # (from report/)
│   └── tf_golden/                    # (from tf_np_golden_gen/)
│
├── tests/                            # Tests (needs expansion)
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_training.py
│   └── ...
│
├── benchmarks/
├── colabs/
├── demos/
└── scripts/                          # Shell scripts & env setup
    ├── compile_report.sh
    ├── create_new_exploration.sh
    └── dev_env_setup/                # (from dev_env_setup_scripts/)
```

---

## Prioritized Action Items

### Phase 1 — Quick Wins (Low Risk)
1. ~~Merge `quantization/` and `quantizations/` into one directory~~ **DONE**
2. Add `__init__.py` to `variations/`, `train_variations/`, `utils/`
3. Move standalone tool scripts from root to `tools/`
4. Move shell scripts from root to `scripts/`

### Phase 2 — Config Cleanup (Medium Risk)
5. Group `explorations/` and `hp_searches/` under `configs/`
6. Standardize `data/` preparation interface (common entry point pattern)
7. Make `data/template/utils/` an importable package, remove duplicated scripts

### Phase 3 — Core Refactoring (Higher Risk, Higher Reward)
8. Extract data loading, checkpointing, and evaluation from `train.py` into `training/` submodules
9. Split `sample.py` into tokenizer management + sampling logic
10. Introduce nested sub-configs in `gpt_conf.py`
11. Group `train_args.py` arguments into category-based modules
12. Move research/experiment directories under `research/`

### Phase 4 — Quality & Testing
13. Expand `tests/` with unit tests for each variation module
14. Add integration tests for the train-sample loop
15. Consider adding a `Makefile` or `pyproject.toml` with standard targets (`make test`, `make lint`, etc.)

---

## Migration Notes

- Each phase can be done independently without breaking the others.
- Import paths will change; use a deprecation period with re-exports from old paths if needed.
- The `data/` directory changes can be done dataset-by-dataset.
- `train.py` refactoring is the highest-value change but requires the most careful testing.
