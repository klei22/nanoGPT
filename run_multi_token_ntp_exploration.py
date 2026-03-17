# run_multi_token_ntp_exploration.py
# ======================================================================
# Runner for multi-token NTP explorations.
# Uses the same YAML-driven exploration system as run_from_yaml.py
# but calls train_multi_token_ntp.py instead of train.py.
# ======================================================================
#
# Usage:
#   python run_multi_token_ntp_exploration.py \
#     --yaml explorations/multi_token_ntp_comparison.yaml \
#     --output_dir out/multi_token_ntp
#
# Or with overrides:
#   python run_multi_token_ntp_exploration.py \
#     --yaml explorations/multi_token_ntp_comparison.yaml \
#     --output_dir out/multi_token_ntp \
#     --override_args max_iters=5000 batch_size=32
# ======================================================================

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import yaml
from rich import print
from rich.console import Console
from rich.table import Table

METRICS_FILENAME = "best_val_loss_and_iter.txt"
METRIC_KEYS = ["best_val_loss", "best_val_iter", "best_tokens", "num_params"]


def parse_override_args(arg_list):
    if not arg_list:
        return {}
    overrides = {}
    for item in arg_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            parsed = yaml.safe_load(value)
        except Exception:
            parsed = value
        overrides[key.strip()] = parsed
    return overrides


def read_metrics(out_dir):
    path = Path(out_dir) / METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    line = path.read_text().strip()
    parts = [p.strip() for p in line.split(",")]
    casts = [float, int, int, int]
    return {k: typ(v) for k, typ, v in zip(METRIC_KEYS, casts, parts[: len(METRIC_KEYS)])}


def completed_runs(log_file):
    if not log_file.exists():
        return set()
    runs = set()
    for doc in yaml.safe_load_all(log_file.open()):
        if doc and "formatted_name" in doc:
            runs.add(doc["formatted_name"])
    return runs


def append_log(log_file, name, combo, metrics):
    entry = {"formatted_name": name, "config": combo, **metrics}
    with log_file.open("a") as f:
        yaml.safe_dump(entry, f, explicit_start=True)


def build_command(combo):
    """Build command for train_multi_token_ntp.py."""
    train_script = combo.pop("_train_script", "train_multi_token_ntp.py")
    cmd = ["python3", train_script]
    for k, v in combo.items():
        if k.startswith("_"):
            continue
        if isinstance(v, bool):
            cmd.append(f"--{'' if v else 'no-'}{k}")
        elif isinstance(v, list):
            cmd.append(f"--{k}")
            cmd.extend(str(x) for x in v)
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def expand_named_groups(yaml_data):
    """
    Expand the named_static_groups + common_group + parameter_groups
    structure into a flat list of config dicts (one per experiment).
    """
    # Parse named static groups into a lookup
    named_groups = {}
    for grp in yaml_data.get("named_static_groups", []):
        name = grp["named_group"]
        settings = {k: v[0] if isinstance(v, list) and len(v) == 1 else v
                     for k, v in grp.items() if k != "named_group"}
        # Also handle named_group_settings sub-dict
        if "named_group_settings" in grp:
            for k, v in grp["named_group_settings"].items():
                settings[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
            del settings["named_group_settings"]
        named_groups[name] = settings

    # Common group: flatten single-element lists
    common = {}
    for k, v in yaml_data.get("common_group", {}).items():
        common[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

    # Expand parameter groups
    configs = []
    for pg in yaml_data.get("parameter_groups", []):
        config = dict(common)  # start with common

        # Apply named static groups
        for ng_name in pg.get("named_group_static", []):
            if ng_name in named_groups:
                config.update(named_groups[ng_name])

        # Apply direct params (flatten single-element lists)
        for k, v in pg.items():
            if k in ("named_group_static", "named_group_alternates"):
                continue
            config[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

        configs.append(config)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-token NTP exploration experiments"
    )
    parser.add_argument("--yaml", type=str, required=True, help="YAML config file")
    parser.add_argument("--output_dir", type=str, default="out/multi_token_ntp")
    parser.add_argument("--prefix", type=str, default="ntp")
    parser.add_argument("--use_timestamp", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    parser.add_argument("--override_args", type=str, nargs="*")
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    configs = expand_named_groups(yaml_data)
    cli_overrides = parse_override_args(args.override_args)

    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{yaml_path.stem}.yaml"

    console = Console()
    any_failed = False

    for idx, config in enumerate(configs):
        # Apply CLI overrides
        if cli_overrides:
            config.update(cli_overrides)

        # Determine run name from out_dir or index
        run_name = config.get("out_dir", f"{args.prefix}-row{idx}")
        if isinstance(run_name, str) and "/" in run_name:
            run_name = run_name.split("/")[-1]

        if run_name in completed_runs(log_file):
            print(f"[yellow]Skipping already-run:[/] {run_name}")
            continue

        # Set output dir
        if "out_dir" not in config:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.use_timestamp else None
            out_name = f"{timestamp}_{run_name}" if timestamp else run_name
            config["out_dir"] = os.path.join(args.output_dir, out_name)

        config["tensorboard_run_name"] = run_name

        # Display config
        table = Table(title=f"Experiment: {run_name}", show_header=False)
        for k, v in config.items():
            if not k.startswith("_"):
                table.add_row(k, str(v))
        console.print(table)

        # Build and run
        cmd = build_command(config.copy())
        print(f"[cyan]Running:[/] {' '.join(cmd)}")

        if args.dry_run:
            print("[cyan]Dry run — skipping execution.[/]")
            continue

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            subprocess.run(cmd, check=True, env=env)
            ok = True
        except subprocess.CalledProcessError:
            print(f"[red]Process exited with error for:[/] {run_name}")
            ok = False
            any_failed = True

        # Record metrics
        try:
            metrics = read_metrics(str(config["out_dir"]))
        except Exception:
            metrics = {k: float("nan") for k in METRIC_KEYS}

        append_log(log_file, run_name, config, metrics)
        print(f"[green]Completed:[/] {run_name}")
        print()

    # Summary
    if log_file.exists():
        print(f"\n[bold]Results log:[/] {log_file}")
        results = []
        for doc in yaml.safe_load_all(log_file.open()):
            if doc and "formatted_name" in doc:
                results.append(doc)

        if results:
            summary = Table(title="Multi-Token NTP Comparison")
            summary.add_column("Run")
            summary.add_column("NTP Tokens")
            summary.add_column("Best Val Loss")
            summary.add_column("Best Iter")
            summary.add_column("Params")
            for r in results:
                ntp = str(r.get("config", {}).get("ntp_tokens", "?"))
                summary.add_row(
                    r["formatted_name"],
                    ntp,
                    f"{r.get('best_val_loss', float('nan')):.6f}",
                    str(r.get("best_val_iter", "?")),
                    f"{r.get('num_params', 0):,}",
                )
            console.print(summary)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
