# nanoGPT Repository Reorganization Plan

## Analysis Summary

### Current State (294 .py files, 230 .sh files, 27 top-level dirs)

The repo has grown organically and has several ergonomic/consistency issues that
hurt tab-completion and discoverability.

---

## Problems Identified

### P1: Prefix collisions kill tab-completion

In zsh/vim, typing `tra<tab>` matches **3 different things** at root level:
- `train.py`, `train_args.py`, `train_mezo.py`, `train_recurrent.py`
- `train_variations/`

Typing `vi<tab>` matches:
- `view_hp_log.py`, `view_model_stats.py`
- `variations/`

Typing `qu<tab>` matches:
- `quantization/`
- `quantizations/`

Typing `sa<tab>` matches:
- `sample.py`
- `sample_variations/`

These prefix collisions mean you must type many more characters before
tab-completion resolves uniquely.

### P2: Two quantization directories (`quantization/` vs `quantizations/`)

Identical prefix. You have to type `quantization<tab>` and then manually pick
the right one. They serve overlapping purposes.

### P3: Inconsistent suffix convention (`_variations` vs `_variants`)

- `variations/` uses `*_variations.py` (15 files) but also has
  `output_vector_variants.py` (1 outlier)
- `train_variations/` uses `*_variants.py` (4 files)
- Two different conventions for the same concept

### P4: Root-level file sprawl

17 Python files at root. Many are utilities that don't need to be top-level:
- `analyze_numerical_output.py` - analysis tool
- `colorize_dataset.py` - data visualization
- `hyperparam_search.py` - search orchestration
- `model_merge.py` - model utility
- `plot_view.py` - plotting
- `run_exploration_monitor.py` - experiment monitoring
- `shared_param_utils.py` - shared utilities
- `update_instances.py` - utility
- `view_hp_log.py` - log viewer
- `view_model_stats.py` - stats viewer

These make tab-completion noisier for the core files (`train.py`, `model.py`,
`sample.py`, `gpt_conf.py`, `train_args.py`).

### P5: Directory name inconsistencies

- `HW/` is UPPERCASE (everything else is lowercase snake_case)
- `logging/` shadows Python's stdlib `logging` module
- `huggingface_model/` - singular, could be confused
- `dev_env_setup_scripts/` - very long directory name
- `tf_np_golden_gen/` - cryptic abbreviation
- `util_factorization/` - inconsistent with `utils/`

### P6: Missing `__init__.py` files

Only `benchmarks/` has an `__init__.py`. Other directories used as packages
(`variations/`, `train_variations/`, `utils/`, `quantization/`) lack them.
This works with implicit namespace packages but is inconsistent and can cause
import issues.

### P7: `sample_variations/` has only 1 file

An entire directory for a single file (`numerical_multicontext.py`).

---

## Reorganization Principles

1. **Unique prefixes** - every top-level entry should be tab-completable in
   <=4 characters
2. **Consistent suffixes** - pick one convention and stick to it
3. **Core files obvious** - the 3-5 most important files should stand out
4. **No stdlib shadows** - don't shadow Python module names
5. **Lowercase snake_case everywhere** - no exceptions
6. **Minimal disruption** - changes should be mechanical and scriptable

---

## Multi-Step Plan (Lightest Changes First)

Each step is self-contained. Complete one, verify, propagate changes, then move
to the next. Steps are ordered from lowest-risk/highest-impact to
highest-risk/lowest-impact.

### Step 1: Fix the one-character consistency issues (zero import impact)

**Changes:**
- Rename `HW/` → `hw/`
- Fix `output_vector_variants.py` → `output_vector_variations.py` in
  `variations/`
- Fix any double extensions (e.g., `.sh.sh` files)

**Propagation:** grep for all references to these names in `.py`, `.sh`,
`.yaml`, `.json`, `.yml`, `.md` files and update them.

**Risk:** Very low. `HW/` is not imported by core code. The variants→variations
rename is a single file with likely few importers.

---

### Step 2: Merge duplicate directories

**Changes:**
- Merge `quantizations/` contents into `quantization/ptq/` (or just
  `quantization/`)
- Move `sample_variations/numerical_multicontext.py` into `variations/` (or
  into a `sampling/` subdirectory within `variations/`)

**Propagation:** Update imports and any shell scripts referencing moved files.

**Risk:** Low. These are peripheral modules with few dependents.

---

### Step 3: Rename `logging/` to avoid stdlib shadow

**Changes:**
- Rename `logging/` → `logs/` (or `log_utils/`)

**Propagation:** Update any references in shell scripts or configs.
`logging/` only contains `start_tensorboard.sh` and `view_csv_logs.py` — no
Python imports reference it as a package.

**Risk:** Very low. No Python imports affected.

---

### Step 4: Standardize suffix convention across variation directories

**Changes:**
- Decide on ONE suffix: `_variations` (already dominant: 15 files use it vs 4
  using `_variants`)
- In `train_variations/`:
  - `distillation_loss_variants.py` → `distillation_loss_variations.py`
  - `eta_variants.py` → `eta_variations.py`
  - `loss_variants.py` → `loss_variations.py`
  - `optimizer_variants.py` → `optimizer_variations.py`

**Propagation:** Update all imports in `train.py`, `model.py`, `sample.py`,
and any other files that import from `train_variations/`.

**Risk:** Medium. These are actively imported by core files. Must update
imports carefully.

---

### Step 5: Move root-level utility scripts into appropriate directories

**Changes:**
- `view_hp_log.py` → `utils/view_hp_log.py`
- `view_model_stats.py` → `utils/view_model_stats.py`
- `plot_view.py` → `utils/plot_view.py`
- `analyze_numerical_output.py` → `analysis/analyze_numerical_output.py`
- `colorize_dataset.py` → `utils/colorize_dataset.py`
- `model_merge.py` → `utils/model_merge.py`
- `shared_param_utils.py` → `utils/shared_param_utils.py`
- `update_instances.py` → `utils/update_instances.py`
- `run_exploration_monitor.py` → `utils/run_exploration_monitor.py`
- `hyperparam_search.py` → `hp_searches/hyperparam_search.py`

**After cleanup, root level contains only:**
```
train.py          (core: training)
train_args.py     (core: argument parsing)
model.py          (core: model definition)
gpt_conf.py       (core: configuration)
sample.py         (core: inference/generation)
train_mezo.py     (alt training method)
train_recurrent.py (alt training method)
```

**Propagation:** Update any scripts, configs, or READMEs that reference moved
files. Update any relative imports if these scripts import sibling modules.

**Risk:** Medium. These are standalone scripts mostly run directly, so imports
into them (not from them) need updating. Shell scripts in `demos/` frequently
reference root-level files.

---

### Step 6: Shorten/normalize long directory names

**Changes:**
- `dev_env_setup_scripts/` → `setup/`
- `tf_np_golden_gen/` → `golden_tests/` (or `reference_tests/`)
- `optimization_and_search/` → `optimization/`
- `huggingface_model/` → `huggingface/`
- `accelergy-timeloop-infrastructure/` → leave as-is (submodule/external)
- `util_factorization/` → fold into `utils/factorization/`

**Propagation:** Update references in shell scripts, configs, CI workflows,
and any imports.

**Risk:** Medium. CI workflows in `.github/workflows/` may reference these
directory names.

---

### Step 7: Add `__init__.py` files to all package directories

**Changes:**
Add empty `__init__.py` to:
- `variations/`
- `train_variations/`
- `utils/`
- `quantization/`
- `analysis/`
- `distillation/`
- `initializations/`

**Propagation:** None needed — additive change only.

**Risk:** Very low. Makes packages explicit and consistent.

---

### Step 8: (Optional) Consolidate `distillation/` into `train_variations/`

**Changes:**
- Move `distillation/angle_optimization.py` and
  `distillation/get_feature_vectors.py` into
  `train_variations/distillation/`

**Propagation:** Update imports.

**Risk:** Low.

---

## Tab-Completion Before & After

### Before (root level):
```
a<tab>  → analyze_numerical_output.py, analysis/, accelergy-...
c<tab>  → colorize_dataset.py, colabs/
d<tab>  → data/, demos/, dev_env_setup_scripts/, distillation/, documentation/
h<tab>  → hp_searches/, huggingface_model/, hyperparam_search.py, HW/
m<tab>  → model.py, model_merge.py
p<tab>  → plot_view.py
q<tab>  → quantization/, quantizations/
r<tab>  → report/, run_exploration_monitor.py
s<tab>  → sample.py, sample_variations/, shared_param_utils.py
t<tab>  → tests/, tf_np_golden_gen/, train.py, train_args.py, train_mezo.py,
          train_recurrent.py, train_variations/
u<tab>  → update_instances.py, util_factorization/, utils/
v<tab>  → variations/, view_hp_log.py, view_model_stats.py
```

### After (root level, all steps applied):
```
a<tab>  → analysis/                    (unique at 'a')
b<tab>  → benchmarks/                  (unique at 'b')
c<tab>  → colabs/                      (unique at 'c')
d<tab>  → data/, demos/, distillation/, documentation/
e<tab>  → explorations/, exutorch/
g<tab>  → golden_tests/, gpt_conf.py   (unique at 'gp' vs 'go')
h<tab>  → hp_searches/, huggingface/, hw/
i<tab>  → initializations/             (unique at 'i')
l<tab>  → logs/                        (unique at 'l')
m<tab>  → model.py                     (unique at 'm')
o<tab>  → optimization/               (unique at 'o')
q<tab>  → quantization/               (unique at 'q')
r<tab>  → report/                     (unique at 'r')
s<tab>  → sample.py, setup/           (unique at 'sa' vs 'se')
t<tab>  → tests/, train.py, train_args.py, train_mezo.py,
          train_recurrent.py, train_variations/
u<tab>  → utils/                      (unique at 'u')
v<tab>  → variations/                 (unique at 'v')
```

Key improvements:
- `m<tab>` → `model.py` immediately (was 2 matches)
- `q<tab>` → `quantization/` immediately (was 2 matches)
- `v<tab>` → `variations/` immediately (was 3 matches)
- `s<tab>` reduced from 3 matches to 2
- `u<tab>` → `utils/` immediately (was 3 matches)
- `r<tab>` → `report/` immediately (was 2 matches)
- Core files (`train.py`, `model.py`, `sample.py`) stand out clearly

---

## Execution Checklist

For each step:
1. [ ] Make the rename/move
2. [ ] `grep -r` for all old references across `.py`, `.sh`, `.yaml`, `.json`,
       `.yml`, `.md` files
3. [ ] Update all references found
4. [ ] Run existing tests (`python -m pytest tests/` or equivalent)
5. [ ] Verify core imports work: `python -c "from variations import ..."`
6. [ ] Commit with descriptive message
7. [ ] Move to next step
