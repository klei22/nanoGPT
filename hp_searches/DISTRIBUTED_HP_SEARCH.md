# Distributed Greedy Hyperparameter Search Design

This note describes how to distribute `hyperparam_search.py` evaluations across
multiple machines while reusing the remote execution framework already present
under `optimization_and_search/nsga_search/`.

## Current control flow

`hyperparam_search.py` is a sequential greedy controller:

1. Load a baseline YAML and any existing `sweep_log.yaml` state.
2. Measure or resume the current baseline metrics.
3. For each outer growth step, generate candidate configs from `param_names`,
   `increments`, `iterations`, and special `n_layer` duplication rules.
4. Evaluate every candidate, and every seed for that candidate, by calling one
   trial runner.
5. Compute averaged score/RankMe/AReQ, cost deltas, and efficiency.
6. Pick the best positive-efficiency candidate, mutate the baseline config, and
   append the iteration to the YAML log.

Only step 4 needs distribution. Candidate generation, efficiency calculation,
baseline mutation, and logging can remain centralized on the coordinator.

## Reusable pieces already in the repo

The NSGA search code already has the building blocks needed for distributed
execution:

- `optimization_and_search/nsga_search/remote_trainer.py`
  - SSH host management through Fabric.
  - Optional `git pull` on each host.
  - Round-robin splitting of a list YAML into per-host YAML slices.
  - Remote launch under a conda environment.
  - PID files, remote run logs, exit-code files, polling, wait, kill, and result
    fetching.
- `optimization_and_search/run_from_yaml.py`
  - Reads a list of configs.
  - Runs `train.py` once per config.
  - Writes one multi-document YAML result file.

This is enough for embarrassingly parallel batches, which is exactly what each
`hyperparam_search.py` iteration produces.

## Recommended architecture

### 1. Factor trial evaluation behind a batch interface

Keep the existing local single-trial runners for backwards compatibility, but
add an interface that evaluates a batch of trial configs and returns metrics in
input order:

```python
@dataclass
class TrialSpec:
    trial_id: str
    candidate_id: str
    param: str
    value: Any
    seed: int
    config: dict[str, Any]

class TrialBackend(Protocol):
    def evaluate(self, trials: list[TrialSpec]) -> dict[str, TrialMetrics]: ...
```

Implement two backends:

- `LocalTrialBackend`: loops over `trials` and calls the existing
  `run_trial_inproc` or `run_trial_subproc`.
- `RemoteYamlTrialBackend`: writes `trials` as a list YAML, delegates it to the
  shared remote launcher, waits, fetches result YAML, and maps rows back by
  `trial_id`.

The greedy optimizer should generate all seed-level trials for one outer
iteration before evaluation. After `backend.evaluate(...)` returns, the existing
averaging and efficiency code can group by `candidate_id` and choose a winner.

### 2. Generalize the NSGA remote runner rather than duplicating it

`RemoteTrainer.submit_job()` currently assumes the remote command is
`optimization_and_search/run_from_yaml.py` plus NSGA-specific paths. Make this a
small generic base class or add a generic method:

```python
def submit_yaml_job(
    self,
    *,
    path_to_yaml: str,
    remote_work_dir: str,
    dir_name: str,
    command_template: str,
    result_filename: str,
    conda_env: str,
) -> bool:
    ...
```

`command_template` can receive `{yaml}`, `{output_dir}`, `{prefix}`, and other
launcher fields. NSGA can keep using the same method with
`optimization_and_search/run_from_yaml.py`; HP search can use a new worker entry
point, for example `hp_searches/run_trials_from_yaml.py`.

The existing `RemoteJob`, heartbeat, `poll_jobs()`, `wait_for_all()`,
`kill_all()`, host connectivity checks, and result download behavior should be
left in one shared implementation.

### 3. Add a small HP-search worker script

A worker script should do only seed-level trial execution, not greedy search:

```bash
python -u hp_searches/run_trials_from_yaml.py \
  --yaml /remote/run/trials.host0.yaml \
  --output_dir /remote/run \
  --results_file hp_trials.yaml \
  --spawn_subprocess
```

Each input item should include metadata plus a full training config:

```yaml
- trial_id: iter3:cand7:seed0
  candidate_id: iter3:cand7
  param: n_embd
  value: 384
  seed: 1337
  config:
    dataset: minipile
    n_embd: 384
    out_dir: /remote/run/iter3_cand7_seed0
```

Each output YAML document should include the same identifiers and the full metric
set currently represented by `TrialMetrics`:

```yaml
---
trial_id: iter3:cand7:seed0
candidate_id: iter3:cand7
status: completed
metrics:
  loss: 3.42
  params: 12400000
  best_iter: 5000
  peak_torch_allocated_mb: 3210
  peak_torch_reserved_mb: 4096
  peak_process_gpu_mb: 4500
  iter_latency_ms: 82.1
  rankme: .nan
  areq: .nan
```

The worker can reuse `dict_to_cli()`, `run_trial_subproc()`, and
`_parse_best_metrics_file()` from `hyperparam_search.py` so metric parsing stays
consistent with local runs.

### 4. Add coordinator CLI flags to `hyperparam_search.py`

Suggested flags:

```text
--distributed                         enable remote batch backend
--hosts_file PATH                     YAML list of hosts
--ssh_user USER
--ssh_key PATH
--remote_work_dir PATH                repo path on each worker
--conda_env NAME
--remote_run_dir NAME                 e.g. hp_search_<timestamp>
--remote_poll_interval SECONDS
--remote_timeout SECONDS
--remote_git_pull                     run git pull on every worker first
--remote_keep_artifacts               do not delete fetched per-host slices/logs
```

When `--distributed` is set, force subprocess-style training on workers. Inproc
training is useful locally, but subprocess isolation is safer for a long-lived
remote worker that executes many CUDA trials.

### 5. Preserve resume semantics

The coordinator should continue to own `sweep_log.yaml` and write it only after a
complete outer iteration. For restartability:

- Write the generated trial batch for each outer iteration next to the log, for
  example `sweep_log.iter003.trials.yaml`.
- Include `trial_id`, `candidate_id`, param label, value, seed, and config in the
  candidate's log entry.
- If the coordinator restarts and a remote result file exists, load completed
  `trial_id`s and submit only missing trials.
- If a trial fails, record the failure in the candidate's `seeds` list and omit
  that candidate from winner selection unless all required seed runs completed.

### 6. Result mapping details

`optimization_and_search/run_from_yaml.py` currently uses `idx` for NSGA result
alignment. The HP-search worker should use explicit string IDs instead:

- `trial_id` uniquely identifies a seed-level training run.
- `candidate_id` groups seed runs into a candidate average.
- The coordinator keeps `candidate_id -> candidate metadata` in memory and in
  the iteration trial YAML.

This avoids depending on row order after remote splits and lets the fetch layer
merge partial host outputs deterministically.

### 7. Why not distribute the whole greedy loop?

The greedy loop is inherently sequential across outer iterations because each
winner mutates the next baseline. Distributing the controller would create log
locking and conflict problems without much benefit. The high-cost work is the
candidate/seed evaluation inside each iteration, so a single coordinator plus
many stateless remote workers is the simpler and more robust design.

## Minimal implementation sequence

1. Add `hp_searches/run_trials_from_yaml.py` that executes trial-list YAML and
   writes trial-result YAML.
2. Add a generic `submit_yaml_job()` method to `RemoteTrainer`; refactor current
   `submit_job()` to call it with the existing NSGA command.
3. Add `RemoteYamlTrialBackend` to `hyperparam_search.py` or a new helper module.
4. Refactor `_evaluate()` so it records candidate metadata first, evaluates all
   generated seed trials as one batch, then computes the existing candidate
   dictionaries from returned metrics.
5. Add CLI flags and document an example distributed HP-search invocation.
6. Keep the current local behavior as the default path and add a dry-run mode
   that prints the trial batch and remote command without launching SSH jobs.

## Example distributed invocation

```bash
python3 hyperparam_search.py \
  --orig_settings ./hp_searches/shakespeare_char.yaml \
  --param_names n_layer n_head n_embd mlp_size_layerlist \
  --increments 1 1 16 16 \
  --iterations 1 \
  --random_iterations 3 \
  --num_iterations 100 \
  --nlayer_dup_mode dup_each \
  --results_file sweep_log.yaml \
  --distributed \
  --hosts_file optimization_and_search/host_configs/hosts.yaml \
  --ssh_user xinting \
  --ssh_key ~/.ssh/id_rsa \
  --remote_work_dir /home/xinting/Evo_GPT \
  --conda_env reallmforge \
  --remote_git_pull
```

With four hosts, one outer step that produces 20 candidates and 3 seeds per
candidate becomes 60 independent training trials. The coordinator writes a
60-row trial YAML, the remote framework splits it round-robin across hosts, each
host runs its 15 trials sequentially, and the coordinator merges results before
choosing the next greedy step.

## Main risks and mitigations

- **Uneven trial durations:** round-robin splitting is simple but can leave one
  host with slow models. A later improvement is a pull-based queue, but the
  round-robin implementation is enough to reuse the existing framework quickly.
- **Remote environment drift:** keep `perform_git_pull()` and log Python/conda
  diagnostics as the current launcher does.
- **Partial failures:** fail only the affected candidates, record missing trials,
  and allow resubmission by `trial_id`.
- **Metric schema drift:** centralize metric parsing in the HP worker by reusing
  the same parser used by `run_trial_subproc()`.
