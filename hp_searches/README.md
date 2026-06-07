# Hyperparameter Searches

This is a folder for hyperparameter searches, currenlty we have a greedy search
algorithm which allows us to inspect the balance of parameters when growing a
model from a baseline.

## Usage

1. Create a baseline.yaml file and store in the hp_searches/ dir

```yaml
# static
dataset: "minipile"
use_rotary_embeddings: True
use_abs_pos_embeddings: False
use_flash_lobo: true
use_flash_lobo_per_head: true
use_concat_heads: False
attention_variant: "infinite"
max_iters: 5000
eval_interval: 5000
compile: True
batch_size: 64
block_size: 256
n_cproj: 1
# changeable things
n_layer: 1
n_head: 1
n_embd: 32
mlp_size: 32
n_qk_head_dim: 32
n_v_head_dim: 32
flash_lobo_log_const: 0.1
```

2. Create a bash script for running this and store in the hp_searches/ dir

Note: list parameter names in the same order as their min step size:
```bash
#!/bin/bash
# lobo_attnhead_search.sh

python3 hyperparam_search.py \
  --orig_settings ./hp_searches/lobo_attnhead_search.yaml \
  --param_names \
        n_layer \
        n_head \
        n_cproj \
        n_embd \
        mlp_size \
        n_qk_head_dim \
        n_v_head_dim \
  --increments \
        1 \
        1 \
        1 \
        32 \
        32 \
        32 \
        32 \
  --random_iterations 1 \
  --iterations 1 \
  --num_iterations 20000 \
  --efficiency_target params \
  --override max_iters=20000 batch_size=64 \
  --results_file results.yaml
```

- `random_iterations` is the number of trials to average for per hp, e.g. n_layer
maybe we can try with 3 random seeds, and get the average to try to fish through
noise.
- `iterations` is the depth of the search per parameter (e.g. try adding 1
    n_layer then trying adding a second n_layer, and see which is best)
- `num_iterations` is the number of growth steps
- `override` don't use this at first, but allows you to manually override
    settings for any hp_search already started (just stop and resume with these
    overrides to for example increase the max_iters, and to unblock the model
    when delta score gets too close to noise levels)
 - `results_file` where to store results for viewing with `view_hp_log.py`
  - `efficiency_target` chooses the cost metric for efficiency: `params`
  (default), `vram` for peak GPU memory, or `iter` for average iteration
  latency.
  - `optimize_target` chooses the improvement objective: `score` (default),
  `rankme`, or `areq`.
  - `optimize_mode` controls direction of optimization for the selected
  target: `max` (default) or `min`.


1. Run bash script from main directory

```bash
bash ./hp_searches/lobo_attnhead_search.sh
```

To quickly exercise the three efficiency metrics, see
`hp_searches/test_efficiency_targets.sh` which runs a tiny search using
the accompanying `efficiency_targets_demo.yaml` baseline.

To demo target optimization with both maximize and minimize modes, run:
- `hp_searches/rankme_target_demo.sh`
- `hp_searches/areq_target_demo.sh`

1. View with `view_hp_log.py`

```bash
python view_hp_log.py results.yaml
```

Note, this will auto-refresh.

3. Monitor via the above, and update max_iters as necessary.

The hyperparameter_search.py has override features, useful for changing training
settings needed as the model grows, e.g. max_iters, learning rate, batch size,
etc.

## Notes And Observations

#### Mitigating Step Noise:

There are a couple ways to improve step noise:

1. Increase the training data, e.g. max_iters (stay less than 1 epoch)
2. Increase the # random iterations to average (e.g. if data is limited)

### Step Settings Considerations:

1. Too large a step size and we can miss sub-optimizations
2. Too small a step size and we encounter too much noise
3. Increase iteration number -- try multiples of the step size, hoping to get
   past noise. Too night an iteration number is wasteful due to reduction in the
   % correct naturally for parameter even in the right direction.

### Training Step Considerations

1. too many training steps, too minimal delta param.
2. too few training steps, too much noise

There are some "humps" for learned parameters, e.g. absolute position encodings,
which typically resolve after 10,000 iterations or so. Currently we try to set a
at 5000 iteration count (with eval_interval at 5000) then move to 20_000 once we
hit a snag.

## Background

###  Why Grow a Model?

We find that we have a large number of hyperparameters, and the hyperparameters
have a strong interdependency on each other.

This means that we really should continually test compatbility of different
techniques during the search.

One way the team is exploring to do this, is to start a small model, and
continually take a step in the most parameter efficient direction.

In this case, we can ensure compatbility of the different parameters (as they
are continually co-tested), and each step will be the greatest increase in
capability per parameter.

### Limitations of Target Optimization

Our prior attempts at optimization include:

1. Optimizing hyperparameters for set # of total parameters:
    a. the hyperparameter space was still too large
    b. we had a danger that the model would simply increase its number of parameters to fit in the maximum space, making it hard to interpret the ultimate shape.

2. Optimizing for parameter efficiency
    a. the most efficient parameters for % of next token correctness were the initial parameters, so this would tend our models to go to zero-parameters.

### How Growing the Model Addresses 1 and 2

*Continual Pressure for Parameter Efficiency*

In this way we provide, "pressure" for the model to balance its parameters at
each stage of growth.

Essentially we have a gradient at each of the steps for how to grow the model,
and try to normalize by the number of additional parameters required to obtain
that improvement (the most cost efficient step).

There are other types of cost efficiency we can also explore, as well as
specific task targets, that make this an interesting framework to develop.

### Supported Metrics and Planned Extensions

Current optimization targets are score (`1/exp(val_loss)`), RankMe, and
AReQ. Each target can be optimized in either maximize or minimize mode.

Later we hope to also include summarization tasks, translation tasks, etc. so
and see what model shapes result from each of these.

Also the "Cost" can be altered as well.

## Next Steps

### Means to increase maxiters automatically upon no feasible direction
We might program a means to have this occur automatically (step size for
max_iters specifically) if we have negative or zero for each of the parameters
directions

### Curve Fitting on Ratios

With the resulting data, we can try to find the curve of parametesr per mlp
size, nad vest val loss vs different characteristics, and ratios of differnet
ones via curve fitting. This could yield insights on balancing parameters
especially towards small language models.

## Distributed / multi-machine candidate evaluation

`hyperparam_search.py` can keep the greedy controller local while sending the
expensive candidate seed trials for each outer iteration to multiple machines.
The search is still a greedy hp_search, not DDP: each remote process trains a
complete candidate trial independently, writes metrics, and returns them to the
controller. The controller then averages seeds, computes efficiency, chooses the
best positive-efficiency step, updates `results_file`, and starts the next outer
iteration.

A ready-to-edit demo is included:

- `hp_searches/multimachine_efficiency_demo.yaml` — a small CUDA baseline using
  `shakespeare_char`, rotary embeddings, no absolute position embeddings, and an
  infinite-attention shape that is cheap enough for a smoke/demo search.
- `hp_searches/multimachine_efficiency_demo.sh` — a wrapper that shards each
  candidate/seed trial across `HP_SEARCH_HOSTS` and exposes common settings as
  environment variables.

### One-time remote setup

Each machine listed in `HP_SEARCH_HOSTS` must be reachable from the controller
with SSH and must already have:

1. The same repository checkout, preferably at the same path on every host.
2. The same git commit checked out as the controller.
3. The training data prepared at the same relative path used by the YAML
   baseline, for example `data/shakespeare_char`.
4. A working Python/conda environment with the repo dependencies installed.
5. GPU/runtime compatibility with the YAML (`device: "cuda"` in the demo). If a
   host should run CPU trials, change the YAML before launching.

Example remote preparation sketch:

```bash
ssh ubuntu@10.0.0.11
cd /home/ubuntu/Evo_GPT
git fetch origin
git checkout <same-branch-or-commit-as-controller>
conda activate reallmforge
python -c "import torch; print(torch.__version__)"
```

Repeat that for every host, or automate it with your cluster tooling.

### Running the demo

Run the wrapper from the repository root on the controller:

```bash
HP_SEARCH_HOSTS="10.0.0.11 10.0.0.12 10.0.0.13" \
HP_SEARCH_USER=ubuntu \
HP_SEARCH_REMOTE_WORK_DIR=/home/ubuntu/Evo_GPT \
HP_SEARCH_CONDA_ENV=reallmforge \
HP_SEARCH_RESULTS_FILE=multimachine_efficiency_results.yaml \
bash hp_searches/multimachine_efficiency_demo.sh
```

Useful optional environment variables:

- `HP_SEARCH_ORIG_SETTINGS` — alternate YAML baseline path.
- `HP_SEARCH_RANDOM_ITERATIONS` — seeds per candidate. Each candidate/seed pair
  is an independent remote trial, so increasing this also increases available
  parallel work.
- `HP_SEARCH_ITERATIONS` — candidate depths per parameter for a greedy step.
- `HP_SEARCH_NUM_ITERATIONS` — maximum number of outer greedy growth steps.
- `HP_SEARCH_EFFICIENCY_TARGET` — `params`, `vram`, `torch_allocated`,
  `torch_reserved`, `process_gpu`, or `iter`. Use `params` for mixed hardware;
  hardware metrics are only comparable across similar GPUs.
- `HP_SEARCH_MAX_ITERS_INCREASE` — increase `max_iters` when no positive
  efficiency candidate is found.
- `HP_SEARCH_POLL_INTERVAL` — seconds between remote status polls.
- `HP_SEARCH_TIMEOUT` — maximum seconds to wait for one distributed candidate
  batch. The demo defaults to `86400` seconds so a dead machine does not hang the
  controller forever.
- `HP_SEARCH_RUN_DIR_NAME` — namespace under each remote checkout's
  `distributed_trials/` directory.
- `HP_SEARCH_OVERRIDE_CFG` — optional whitespace-separated `KEY=VALUE` overrides
  passed to `--override_cfg`, for example
  `HP_SEARCH_OVERRIDE_CFG="device=cuda:0 dtype=float16 compile=False batch_size=16"`.

Monitor the controller-side log in another terminal:

```bash
python view_hp_log.py multimachine_efficiency_results.yaml
```

### How sharding works

For each outer greedy iteration, the controller builds all candidate configs from
`--param_names`, `--increments`, `--iterations`, and `--random_iterations`. It
then creates one trial record for every candidate/seed pair and round-robin
shards those records across `--distributed_hosts`. On each host, the remote
runner executes its shard sequentially. Every trial gets its own isolated
`out_dir` under:

```text
<remote_work_dir>/distributed_trials/<run_dir_name>/...
```

The remote runner writes `hp_results.yaml` incrementally as trials finish. After
all remote jobs reach a terminal state, the controller fetches these result files
and applies the same local greedy selection logic used by non-distributed
hp_search.

### Mixed GPU types, CUDA device names, and Jetson Orin

The demo sends one YAML config to every host. If host A is a desktop/server GPU
and host B is a Jetson Orin, both hosts still receive the same values for
`device`, `dtype`, `compile`, `batch_size`, and every model hyperparameter. The
default demo uses `device: "cuda:0"`, which is normally valid for a single GPU on
both a desktop CUDA machine and an Orin, but the hardware is not equivalent.

This has a few practical effects:

- **Wall-clock time is limited by the slowest shard.** The controller waits for
  every host's shard before choosing the next greedy step, so an Orin can make
  each distributed batch finish at Orin speed for the trials assigned to it.
- **Use `params` as the efficiency target on mixed hardware.** Parameter deltas
  are hardware-independent, but `iter`, `vram`, `torch_allocated`,
  `torch_reserved`, and `process_gpu` are not comparable between a desktop GPU
  and Jetson unified memory. If you optimize one of those hardware targets, the
  greedy choice can be biased by which host happened to run that candidate.
- **Use lowest-common-denominator runtime settings.** Jetson/aarch64 support for
  BF16, `torch.compile`, and large batches can differ from a desktop GPU. The
  demo therefore uses `dtype: "float16"` and `compile: False`; reduce
  `batch_size` or `block_size` if the Orin runs out of memory.
- **One host cannot currently receive host-specific overrides.** If the desktop
  should use `cuda:1` but the Orin should use `cuda:0`, launch separate
  homogeneous host groups or control visibility on each machine with
  `CUDA_VISIBLE_DEVICES` so the desired accelerator appears as `cuda:0`.

For a mixed desktop CUDA + Orin smoke run, prefer something like:

```bash
HP_SEARCH_HOSTS="desktop-gpu jetson-orin" \
HP_SEARCH_EFFICIENCY_TARGET=params \
HP_SEARCH_OVERRIDE_CFG="device=cuda:0 dtype=float16 compile=False batch_size=16" \
bash hp_searches/multimachine_efficiency_demo.sh
```

For timing/VRAM searches, split the fleet and run one search per hardware class
so every candidate is measured on comparable devices.

### Failure behavior

Distributed hp_search is fault-tolerant enough to preserve the current search
state, but it does not automatically reassign unfinished trials to another host.
Use `HP_SEARCH_TIMEOUT`/`--distributed_timeout` for predictable recovery.

- **Host cannot launch the shard:** submission returns failure, any shard jobs
  launched earlier in that batch are cancelled, `hyperparam_search.py` raises an
  error, and no new greedy step is written. Fix the host list/setup and rerun;
  the controller resumes from the existing `results_file`.
- **Host goes down after launch:** polling logs warnings while the job remains
  non-terminal. If the host comes back before `HP_SEARCH_TIMEOUT`, polling can
  continue and fetch completed results. If the timeout is reached, the batch is
  marked timed out and the controller raises without applying a partial greedy
  step.
- **A remote trial fails but the host remains reachable:** the remote worker
  records that trial as failed and continues with later trials in the same shard.
  During aggregation, candidates missing any required seed are skipped for that
  outer iteration.
- **A result file cannot be fetched:** fetched results from other hosts are still
  read, but candidates that do not have all of their seed runs are skipped. Rerun
  after fixing the host to re-evaluate the same greedy state.
- **Controller interruption:** completed baseline/iteration state is already in
  `results_file`. Restart the same command to resume from the last committed
  greedy iteration.

