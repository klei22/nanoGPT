import json
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
import argparse
import os

import yaml
from rich import print
from rich.console import Console
from rich.table import Table

# Constants
LOG_DIR = Path("exploration_logs")
LOG_DIR.mkdir(exist_ok=True)
METRICS_FILENAME = "best_val_loss_and_iter.txt"
METRIC_KEYS = [
    "best_val_loss",
    "best_val_iter",
    "num_params",
    "better_than_chance",
    "btc_per_param",
    "peak_gpu_mb",
    "iter_latency_avg",
    "tokens_trained",
    "epochs_trained",
    "avg_top1_prob",
    "avg_top1_correct",
    "avg_target_rank",
    "avg_target_left_prob",
    "avg_target_prob",
    "target_rank_95",
    "left_prob_95",
]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments based on a configuration file (JSON or YAML)."
    )
    parser.add_argument(
        '-c', '--config', required=True,
        help="Path to the configuration file."
    )
    parser.add_argument(
        '--config_format', choices=['json', 'yaml'], default='yaml',
        help="Configuration file format (json or yaml)."
    )
    parser.add_argument(
        '-o', '--output_dir', default="out",
        help="Directory to place experiment outputs."
    )
    parser.add_argument(
        '--prefix', default='',
        help="Optional prefix for run names and output directories."
    )
    parser.add_argument(
        '--use_timestamp', action='store_true',
        help="Prepend timestamp to run names and out_dir."
    )
    return parser.parse_args()


def load_configurations(path: str, fmt: str) -> list[dict]:
    """
    Load experiment configurations from a JSON or YAML file.

    Args:
        path: File path.
        fmt: 'json' or 'yaml'.

    Returns:
        A list of configuration dictionaries.
    """
    text = Path(path).read_text()
    if fmt == 'yaml':
        # YAML may contain multiple documents or a single list
        loaded = list(yaml.safe_load_all(text))
        # Flatten if outer list-of-lists
        if len(loaded) == 1 and isinstance(loaded[0], list):
            return loaded[0]
        return loaded
    else:
        return json.loads(text)


RUN_NAME_VAR = "${RUN_NAME}"


def expand_range(val):
    """Expand dicts with 'range' into a list of values."""
    if isinstance(val, dict) and 'range' in val:
        r = val['range']
        start, end = r['start'], r['end']
        step = r.get('step', 1 if isinstance(start, int) else 0.1)
        if isinstance(start, int):
            return list(range(start, end + 1, step))
        count = int(round((end - start) / step)) + 1
        return [start + i * step for i in range(count)]
    return val


def _substitute_run_name(obj, run_name: str):
    """Recursively substitute the run name placeholder inside ``obj``."""
    if isinstance(obj, str):
        return obj.replace(RUN_NAME_VAR, run_name)
    if isinstance(obj, list):
        return [_substitute_run_name(o, run_name) for o in obj]
    if isinstance(obj, dict):
        return {k: _substitute_run_name(v, run_name) for k, v in obj.items()}
    return obj


def generate_combinations(config: dict):
    """Yield all valid parameter combinations for a config dict.

    Supports arbitrarily nested ``parameter_groups``.
    """
    def _expand_base_and_conditionals(cfg: dict):
        # Split plain parameters (base) from conditional specs
        base = {
            k: (expand_range(v) if isinstance(v, dict) and 'range' in v else v)
            for k, v in cfg.items()
            if not (isinstance(v, dict) and 'conditions' in v)
               and k != 'parameter_groups'
        }
        # Ensure each base value is iterable for cartesian product
        base = {k: (v if isinstance(v, list) else [v]) for k, v in base.items()}

        conditionals = {
            k: v for k, v in cfg.items()
            if isinstance(v, dict) and 'conditions' in v
        }
        return base, conditionals

    def _conditions_match(combo: dict, raw_conditions):
        # dict => AND of all pairs; list[dict] => OR across dicts, AND within each dict
        if isinstance(raw_conditions, dict):
            return all(combo.get(k) == v for k, v in raw_conditions.items())
        if isinstance(raw_conditions, list):
            clauses = [d for d in raw_conditions if isinstance(d, dict)]
            if not clauses:
                return False
            return any(all(combo.get(k) == v for k, v in d.items()) for d in clauses)
        return False

    def _apply_conditionals(combo_dict: dict, conditionals: dict):
        valid = [combo_dict]
        for param, spec in conditionals.items():
            next_valid = []
            raw_conditions = spec.get('conditions', {})
            opts = spec.get('options', [])
            options = opts if isinstance(opts, list) else [opts]

            for c in valid:
                if _conditions_match(c, raw_conditions):
                    for opt in options:
                        new_c = dict(c)
                        new_c[param] = opt
                        next_valid.append(new_c)
                else:
                    # If conditions don't match, leave combo unchanged
                    next_valid.append(c)
            valid = next_valid
        return valid

    def recurse(cfg: dict):
        groups = cfg.get('parameter_groups')
        if groups:
            # Coerce to list to handle both single dict and list-of-dicts
            groups_list = groups if isinstance(groups, list) else [groups]
            base_cfg = {k: v for k, v in cfg.items() if k != 'parameter_groups'}
            for grp in groups_list:
                merged = {**base_cfg, **grp}
                yield from recurse(merged)
            return

        base, conditionals = _expand_base_and_conditionals(cfg)
        keys = list(base)
        # itertools.product with zero iterables yields one empty tuple, which is what we want
        for combo in product(*(base[k] for k in keys)):
            combo_dict = dict(zip(keys, combo))
            for final in _apply_conditionals(combo_dict, conditionals):
                yield final

    # Work on a shallow copy to avoid mutating caller's dict
    yield from recurse(dict(config))


def format_run_name(combo: dict, base: str, prefix: str) -> str:
    """
    Create a unique run name from parameter values.
    """
    parts = [str(v) for v in combo.values()
             if not (isinstance(v, str) and RUN_NAME_VAR in v)]
    return f"{prefix}{base}-{'-'.join(parts)}"


def read_metrics(out_dir: str) -> dict:
    """
    Read best_val_loss_and_iter.txt and parse metrics.

    Returns:
        Dict with keys from METRIC_KEYS.
    """
    path = Path(out_dir) / METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    line = path.read_text().strip()
    parts = [p.strip() for p in line.split(',')]

    casts = [
        float,  # best_val_loss
        int,    # best_val_iter
        int,    # num_params
        float,  # better_than_chance
        float,  # btc_per_param
        float,  # peak_gpu_mb
        float,  # iter_latency_avg
        int,    # tokens_trained
        float,  # epochs_trained
        float,  # avg_top1_prob
        float,  # avg_top1_correct
        float,  # avg_target_rank
        float,  # avg_target_left_prob
        float,  # avg_target_prob
        float,  # target_rank_95
        float,  # left_prob_95
    ]
    return {k: typ(v) for k, typ, v in zip(METRIC_KEYS, casts, parts)}


def completed_runs(log_file: Path) -> set[str]:
    """
    Return set of run names already logged in YAML file.
    """
    if not log_file.exists():
        return set()
    runs = set()
    for doc in yaml.safe_load_all(log_file.open()):
        if doc and 'formatted_name' in doc:
            runs.add(doc['formatted_name'])
    return runs


def append_log(log_file: Path, name: str, combo: dict, metrics: dict) -> None:
    """
    Append a YAML entry with run details and metrics.
    """
    entry = {'formatted_name': name, 'config': combo, **metrics}
    with log_file.open('a') as f:
        yaml.safe_dump(entry, f, explicit_start=True)


def build_command(combo: dict) -> list[str]:
    """
    Construct the command-line invocation for train.py.
    """
    cmd = ['python3', 'train.py']
    for k, v in combo.items():
        if isinstance(v, bool):
            cmd.append(f"--{'' if v else 'no-'}{k}")
        elif isinstance(v, list):
            for x in v:
                cmd += [f"--{k}", str(x)]
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def run_experiment(
    combo: dict,
    base: str,
    args: argparse.Namespace
) -> None:
    """
    Execute one experiment combo: skip if done, run train.py, record metrics.
    """
    run_name = format_run_name(combo, base, args.prefix)
    log_file = LOG_DIR / f"{base}.yaml"
    if run_name in completed_runs(log_file):
        print(f"[yellow]Skipping already-run:[/] {run_name}")
        return

    # Prepare output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if args.use_timestamp else None
    out_dir_name = f"{timestamp}_{run_name}" if timestamp else run_name
    combo['out_dir'] = os.path.join(args.output_dir, out_dir_name)

    # Prepare tensorboard run name
    combo['tensorboard_run_name'] = run_name

    # Substitute special run-name token in string parameters
    combo = _substitute_run_name(combo, run_name)

    # Show parameters
    console = Console()
    table = Table("Parameters", show_header=False)
    for k, v in combo.items():
        table.add_row(k, str(v))
    console.print(table)

    # Build and run
    cmd = build_command(combo)
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[red]Process exited with error for run:[/] {run_name}")

    # Read metrics (use existing or nan on failure)
    try:
        metrics = read_metrics(str(combo['out_dir']))
    except Exception:
        metrics = {k: float("nan") for k in METRIC_KEYS}

    append_log(log_file, run_name, combo, metrics)


def main():
    args = parse_args()
    base = Path(args.config).stem
    configs = load_configurations(args.config, args.config_format)

    for cfg in configs:
        for combo in generate_combinations(cfg):
            run_experiment(combo, base, args)


if __name__ == '__main__':
    main()
