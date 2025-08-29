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
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

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
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments based on a configuration file (JSON or YAML).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--config_format",
        choices=["json", "yaml"],
        default="yaml",
        help="Configuration file format (json or yaml).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="out",
        help="Directory to place experiment outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for run names and output directories.",
    )
    parser.add_argument(
        "--use_timestamp",
        action="store_true",
        help="Prepend timestamp to run names and out_dir.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar while running experiments.",
    )
    parser.add_argument(
        "--log_count",
        action="store_true",
        help=(
            "Assume the exploration config is unchanged and count completed runs "
            "using only entries in exploration_logs."
        ),
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


def generate_combinations(config: dict) -> dict:
    """Yield all valid parameter combinations for a config dict.

    This function supports nested ``parameter_groups`` by recursively expanding
    any groups found within the configuration.

    Returns:
        Iterator of parameter-combination dicts.
    """

    def expand(cfg: dict):
        groups = cfg.get("parameter_groups")
        base_items = {k: v for k, v in cfg.items() if k != "parameter_groups"}
        base = {
            k: (expand_range(v) if isinstance(v, dict) and "range" in v else v)
            for k, v in base_items.items()
            if not (isinstance(v, dict) and "conditions" in v)
        }
        base = {k: (v if isinstance(v, list) else [v]) for k, v in base.items()}
        conditionals = {
            k: v
            for k, v in base_items.items()
            if isinstance(v, dict) and "conditions" in v
        }

        combos = []
        keys = list(base)
        for combo_vals in product(*(base[k] for k in keys)):
            combo_dict = dict(zip(keys, combo_vals))
            valid = [combo_dict]
            for param, spec in conditionals.items():
                next_valid = []
                for c in valid:
                    if all(c.get(key) == val for key, val in spec["conditions"]):
                        opts = spec["options"]
                        for opt in (opts if isinstance(opts, list) else [opts]):
                            new = dict(c)
                            new[param] = opt
                            next_valid.append(new)
                    else:
                        next_valid.append(c)
                valid = next_valid
            combos.extend(valid)

        if not groups:
            for c in combos:
                yield c
        else:
            for group in groups:
                for c in combos:
                    merged = {**c, **group}
                    yield from expand(merged)

    yield from expand(config)


def format_run_name(combo: dict, base: str, prefix: str) -> str:
    """
    Create a unique run name from parameter values.
    """
    parts = [str(v) for v in combo.values()]
    return f"{prefix}{base}-{'-'.join(parts)}"


def read_metrics(out_dir: str) -> dict:
    """
    Read best_val_loss_and_iter.txt and parse five metrics.

    Returns:
        Dict with keys: best_val_loss, best_val_iter, num_params,
        better_than_chance, btc_per_param.
    """
    path = Path(out_dir) / METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    line = path.read_text().strip()
    parts = [p.strip() for p in line.split(',')]

    casts = [float, int, int, float, float, float, float]
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
    run_name: str,
    args: argparse.Namespace,
    log_file: Path,
    completed_set: set[str],
) -> None:
    """Execute one experiment combo and record metrics."""
    if run_name in completed_set:
        print(f"[yellow]Skipping already-run:[/] {run_name}")
        return

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.use_timestamp else None
    out_dir_name = f"{timestamp}_{run_name}" if timestamp else run_name
    combo["out_dir"] = os.path.join(args.output_dir, out_dir_name)

    # Prepare tensorboard run name
    combo["tensorboard_run_name"] = run_name

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
        metrics = read_metrics(str(combo["out_dir"]))
    except Exception:
        metrics = {k: float("nan") for k in METRIC_KEYS}

    append_log(log_file, run_name, combo, metrics)
    completed_set.add(run_name)


def main():
    args = parse_args()
    base = Path(args.config).stem
    configs = load_configurations(args.config, args.config_format)

    combos_with_names: list[tuple[dict, str]] = []
    for cfg in configs:
        for combo in generate_combinations(cfg):
            run_name = format_run_name(combo, base, args.prefix)
            combos_with_names.append((combo, run_name))

    log_file = LOG_DIR / f"{base}.yaml"
    completed_set = completed_runs(log_file)
    total = len(combos_with_names)

    if args.progress:
        if args.log_count:
            completed_initial = len(completed_set)
        else:
            completed_initial = sum(
                1 for _, name in combos_with_names if name in completed_set
            )
        remaining = [
            (c, n) for c, n in combos_with_names if n not in completed_set
        ]
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} configs"),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task("configs", total=total, completed=completed_initial)
            for combo, run_name in remaining:
                run_experiment(combo, run_name, args, log_file, completed_set)
                progress.advance(task)
    else:
        for combo, run_name in combos_with_names:
            run_experiment(combo, run_name, args, log_file, completed_set)


if __name__ == '__main__':
    main()

