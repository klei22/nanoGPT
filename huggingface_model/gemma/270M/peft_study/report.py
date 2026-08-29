#!/usr/bin/env python3
"""Create reproducible CSV and Markdown comparisons from study JSON files."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def preferred_metric(metrics: dict) -> tuple[str, float] | None:
    for suffix in ("pass@1,none", "acc_norm,none", "acc,none", "exact_match,none", "perplexity,none"):
        if suffix in metrics and isinstance(metrics[suffix], (int, float)):
            return suffix.split(",")[0], float(metrics[suffix])
    return None


def collect(root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(root.glob("*/evaluation.json")):
        run_dir = result_path.parent
        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        results = json.loads(result_path.read_text()).get("results", {})
        method = metadata.get("method", run_dir.name)
        for task, values in sorted(results.items()):
            metric = preferred_metric(values)
            if metric:
                rows.append({
                    "method": method, "task": task, "metric": metric[0], "value": metric[1],
                    "seed": metadata.get("seed", ""),
                    "trainable_parameters": metadata.get("trainable_parameters", ""),
                    "wall_time_seconds": metadata.get("wall_time_seconds", ""),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_dir", required=True)
    args = parser.parse_args()
    root = Path(args.study_dir)
    rows = collect(root)
    if not rows:
        raise SystemExit(f"No */evaluation.json results found under {root}")
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    methods = sorted({row["method"] for row in rows})
    task_metrics = sorted({(row["task"], row["metric"]) for row in rows})
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["task"], row["metric"]), []).append(row["value"])
    lines = ["# PEFT study report", "", "## Benchmark results", "",
             "| Task / metric | " + " | ".join(methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for task, metric in task_metrics:
        values = []
        for method in methods:
            samples = grouped.get((method, task, metric), [])
            if not samples:
                values.append("—")
            elif len(samples) == 1:
                values.append(f"{samples[0]:.4f}")
            else:
                values.append(f"{statistics.mean(samples):.4f} ± {statistics.stdev(samples):.4f}")
        lines.append(f"| {task} / {metric} | " + " | ".join(values) + " |")
    lines += ["", "## Efficiency", "", "| Method | Trainable parameters | Training wall time (s) |", "|---|---:|---:|"]
    for method in methods:
        method_rows = [item for item in rows if item["method"] == method]
        parameters = next((item["trainable_parameters"] for item in method_rows if item["trainable_parameters"] != ""), "n/a")
        times = [float(item["wall_time_seconds"]) for item in method_rows if item["wall_time_seconds"] != ""]
        time_summary = f"{statistics.mean(times):.2f}" if times else "n/a"
        lines.append(f"| {method} | {parameters} | {time_summary} |")
    lines += ["", "Generated from machine-readable `run_metadata.json` and `evaluation.json` files.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
