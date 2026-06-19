"""Run hyper-parameter search trial shards.

This helper is intentionally small and generic: distributed search controllers write a
YAML list of trial records, then remote workers execute this script to run each
trial sequentially on the assigned machine and persist machine-readable results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

TrialMetrics = Tuple[float, float, int, float, float, float, float, float, float]


def dict_to_cli(config: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for key, value in config.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, bool):
            if value:
                cli.append(f"--{key}")
        elif isinstance(value, list):
            cli.append(f"--{key}")
            cli.extend(map(str, value))
        else:
            cli.extend([f"--{key}", str(value)])
    return cli


def _parse_best_metrics_file(metrics_path: Path) -> TrialMetrics:
    line = [x.strip() for x in metrics_path.read_text().strip().split(",")]
    if len(line) < 21:
        raise ValueError(
            f"Unexpected metric layout in {metrics_path}: expected >=21 columns, got {len(line)}"
        )
    loss = float(line[0])
    best_iter = int(line[1])
    nparam = float(line[3])
    torch_alloc_mb = float(line[6])
    torch_resv_mb = float(line[7])
    process_gpu_mb = float(line[8])
    iter_latency_ms = float(line[9])
    rankme_idx = 20 if len(line) > 21 else 19
    areq_idx = rankme_idx + 1
    rankme = float(line[rankme_idx])
    areq = float(line[areq_idx])
    return (
        loss,
        nparam,
        best_iter,
        torch_alloc_mb,
        torch_resv_mb,
        process_gpu_mb,
        iter_latency_ms,
        rankme,
        areq,
    )


def _run_trial(config: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    script = repo_root / "train.py"
    cmd = [sys.executable, str(script)] + dict_to_cli(config)
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "WORLD_SIZE"}}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=repo_root)
    result: Dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
    }
    if proc.returncode != 0:
        result["status"] = "failed"
        return result

    metrics_path = Path(config.get("out_dir", "out")) / "best_val_loss_and_iter.txt"
    if not metrics_path.is_absolute():
        metrics_path = repo_root / metrics_path
    (
        loss,
        nparam,
        best_iter,
        torch_alloc_mb,
        torch_resv_mb,
        process_gpu_mb,
        iter_latency_ms,
        rankme,
        areq,
    ) = _parse_best_metrics_file(metrics_path)
    result.update(
        {
            "status": "completed",
            "loss": loss,
            "num_params": nparam,
            "best_iter": best_iter,
            "peak_torch_allocated_mb": torch_alloc_mb,
            "peak_torch_reserved_mb": torch_resv_mb,
            "peak_process_gpu_mb": process_gpu_mb,
            "iter_latency_ms": iter_latency_ms,
            "rankme": rankme,
            "areq": areq,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a shard of hp_search trials")
    parser.add_argument("--trials_yaml", required=True)
    parser.add_argument("--results_yaml", required=True)
    parser.add_argument("--repo_root", default=str(REPO_ROOT))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results_path = Path(args.results_yaml)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = results_path.parent / "trial_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    trials: List[Dict[str, Any]] = yaml.safe_load(Path(args.trials_yaml).read_text()) or []
    results: List[Dict[str, Any]] = []
    for trial in trials:
        trial_id = str(trial["trial_id"])
        cfg = dict(trial["config"])
        cfg["out_dir"] = str(output_root / trial_id)
        record = {k: v for k, v in trial.items() if k != "config"}
        record["out_dir"] = cfg["out_dir"]
        try:
            record.update(_run_trial(cfg, repo_root))
        except Exception as exc:  # keep workers alive for later trials in the shard
            record.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        results.append(record)
        results_path.write_text(yaml.safe_dump(results, sort_keys=False))


if __name__ == "__main__":
    main()
