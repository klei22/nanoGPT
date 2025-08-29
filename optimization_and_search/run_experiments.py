import json
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
import argparse
import os
from typing import Optional

import yaml
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

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
    "avg_top1_prob",
    "avg_top1_correct",
    "avg_target_rank",
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
    parser.add_argument(
        '--progress', action='store_true',
        help="Display a rich progress bar while running experiments.",
    )
    parser.add_argument(
        '--simple_log_count', action='store_true',
        help=("Assume the exploration config has not changed and estimate the number of completed configs by counting entries in the log file."),
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


def expand_range(val):
    """
    Expand dicts with 'range' into a list of values.
    """
    if isinstance(val, dict) and 'range' in val:
        r = val['range']
        start, end = r['start'], r['end']
        step = r.get('step', 1 if isinstance(start, int) else 0.1)
        if isinstance(start, int):
            return list(range(start, end + 1, step))
        count = int((end - start) / step) + 1
        return [start + i * step for i in range(count)]
    return val


def _expand_parameter_groups(cfg: dict) -> list[dict]:
    """Recursively expand any nested ``parameter_groups`` entries."""
    groups = cfg.get('parameter_groups')
    base = {k: v for k, v in cfg.items() if k != 'parameter_groups'}
    if not groups:
        return [base]
    expanded = []
    for grp in groups:
        merged = {**base, **grp}
        expanded.extend(_expand_parameter_groups(merged))
    return expanded


def generate_combinations(config: dict) -> dict:
    """Yield all valid parameter combinations for a single config dict."""
    for cfg in _expand_parameter_groups(config):
        base = {
            k: (expand_range(v) if isinstance(v, dict) and 'range' in v else v)
            for k, v in cfg.items()
            if not (isinstance(v, dict) and 'conditions' in v)
        }
        base = {k: (v if isinstance(v, list) else [v]) for k, v in base.items()}
        conditionals = {
            k: v for k, v in cfg.items() if isinstance(v, dict) and 'conditions' in v
        }

        keys = list(base)
        for combo in product(*(base[k] for k in keys)):
            combo_dict = dict(zip(keys, combo))
            valid = [combo_dict]
            for param, spec in conditionals.items():
                next_valid = []
                for c in valid:
                    if all(c.get(key) == val for key, val in spec['conditions']):
                        opts = spec['options']
                        for opt in (opts if isinstance(opts, list) else [opts]):
                            new = dict(c)
                            new[param] = opt
                            next_valid.append(new)
                    else:
                        next_valid.append(c)
                valid = next_valid
            for v in valid:
                yield v


def format_run_name(combo: dict, base: str, prefix: str) -> str:
    """
    Create a unique run name from parameter values.
    """
    parts = [str(v) for v in combo.values()]
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

    casts = [float, int, int, float, float, float, float, float, float, float]
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
    """Construct the command-line arguments for train.py."""
    cmd: list[str] = []
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
    args: argparse.Namespace,
    completed: set[str],
    log_file: Path,
    console: Optional[Console] = None,
) -> bool:
    """Execute one experiment combo.

    Returns True if a new experiment was run, False if it was skipped."""
    run_name = format_run_name(combo, base, args.prefix)
    if run_name in completed:
        print(f"[yellow]Skipping already-run:[/] {run_name}")
        return False

    # Prepare output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if args.use_timestamp else None
    out_dir_name = f"{timestamp}_{run_name}" if timestamp else run_name
    combo['out_dir'] = os.path.join(args.output_dir, out_dir_name)

    # Prepare tensorboard run name
    combo['tensorboard_run_name'] = run_name

    # Show parameters
    display = console or Console()
    table = Table("Parameters", show_header=False)
    for k, v in combo.items():
        table.add_row(k, str(v))
    display.print(table)

    # Build and run
    cmd_args = build_command(combo)
    display.print(f"Running: train.py {' '.join(cmd_args)}")
    if console:
        try:
            import train
            train.main(cmd_args, console=console)
        except Exception as e:
            display.print(
                f"[red]Process exited with an error for run:[/] {run_name} ({e})"
            )
    else:
        try:
            subprocess.run(['python3', 'train.py', *cmd_args], check=True)
        except subprocess.CalledProcessError as e:
            display.print(
                f"[red]Process exited with an error for run:[/] {run_name} ({e})"
            )

    # Read metrics (use existing or nan on failure)
    try:
        metrics = read_metrics(str(combo['out_dir']))
    except Exception:
        metrics = {k: float("nan") for k in METRIC_KEYS}

    append_log(log_file, run_name, combo, metrics)
    completed.add(run_name)
    return True


def main():
    args = parse_args()
    base = Path(args.config).stem
    configs = load_configurations(args.config, args.config_format)
    log_file = LOG_DIR / f"{base}.yaml"
    completed = completed_runs(log_file)

    if args.progress:
        all_runs = []
        for cfg in configs:
            for combo in generate_combinations(cfg):
                run_name = format_run_name(combo, base, args.prefix)
                all_runs.append((combo, run_name))

        total = len(all_runs)
        if args.simple_log_count:
            initial_completed = min(len(completed), total)
        else:
            initial_completed = sum(1 for _, name in all_runs if name in completed)

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task("configs", total=total, completed=initial_completed)
            for combo, _ in all_runs:
                ran = run_experiment(
                    combo, base, args, completed, log_file, console=progress.console
                )
                if ran:
                    progress.advance(task)
    else:
        for cfg in configs:
            for combo in generate_combinations(cfg):
                run_experiment(combo, base, args, completed, log_file)


if __name__ == '__main__':
    main()

