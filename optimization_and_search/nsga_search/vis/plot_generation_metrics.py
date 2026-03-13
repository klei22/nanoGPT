#!/usr/bin/env python3
"""Plot generation-wise metrics across NSGA-II checkpoints."""

import argparse
import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsga2 import Population
from investigate import analyze_population


DEFAULT_METRICS = [
    "accuracy_per_param_mean",
    "accuracy_per_param_max",
]


def _maybe_stat(df: pd.DataFrame, column: str, fn) -> float:
    if column not in df.columns:
        return np.nan
    series = df[column].dropna()
    if series.empty:
        return np.nan
    return float(fn(series))


def compute_generation_metrics(file_name_base: str, start_gen: int, end_gen: int) -> pd.DataFrame:
    """Aggregate population metrics for each generation."""

    records = []
    for gen in range(start_gen, end_gen + 1):
        json_path = f"{file_name_base}{gen}.json"
        if not os.path.exists(json_path):
            print(f"❌ Skipping missing checkpoint: {json_path}")
            continue

        population = Population.load_checkpoint(json_path, from_pkl=False)
        pop_df = analyze_population(population)
        if pop_df.empty:
            print(f"⚠️ No analyzable individuals for generation {gen} (file: {json_path})")
            continue

        val_loss_mean = _maybe_stat(pop_df, "val_loss", np.mean)
        params_mean = _maybe_stat(pop_df, "params", np.mean)
        params_min = _maybe_stat(pop_df, "params", np.min)

        product_mean = np.nan
        product_min = np.nan
        accuracy_params_mean = np.nan
        accuracy_params_max = np.nan
        if {"val_loss", "params"}.issubset(pop_df.columns):
            val_series = pd.to_numeric(pop_df["val_loss"], errors="coerce")
            params_series = pd.to_numeric(pop_df["params"], errors="coerce")
            valid = params_series > 0
            if valid.any():
                params_valid = params_series[valid] / 1e6
                val_valid = val_series[valid]

                product_series = (val_valid * params_valid).replace([np.inf, -np.inf], np.nan).dropna()
                if not product_series.empty:
                    product_mean = float(product_series.mean())
                    product_min = float(product_series.min())

                accuracy_series = np.exp(-val_valid)
                ratio_series = (accuracy_series / params_valid).replace([np.inf, -np.inf], np.nan).dropna()
                if not ratio_series.empty:
                    accuracy_params_mean = float(ratio_series.mean())
                    accuracy_params_max = float(ratio_series.max())

        record = {
            "generation": gen,
            "val_loss_mean": val_loss_mean,
            "val_loss_min": _maybe_stat(pop_df, "val_loss", np.min),
            "energy_mean": _maybe_stat(pop_df, "energy_per_token", np.mean),
            "energy_min": _maybe_stat(pop_df, "energy_per_token", np.min),
            "ttft_mean": _maybe_stat(pop_df, "ttft", np.mean),
            "ttft_min": _maybe_stat(pop_df, "ttft", np.min),
            "params_mean": params_mean,
            "params_min": params_min,
            "params_mean_million": params_mean / 1e6 if not np.isnan(params_mean) else np.nan,
            "params_min_million": params_min / 1e6 if not np.isnan(params_min) else np.nan,
            "val_loss_params_product_mean": product_mean,
            "val_loss_params_product_min": product_min,
            "accuracy_per_param_mean": accuracy_params_mean,
            "accuracy_per_param_max": accuracy_params_max,
            "active_layers_mean": _maybe_stat(pop_df, "n_layers_active", np.mean),
            "active_layers_min": _maybe_stat(pop_df, "n_layers_active", np.min),
            "headcount_mean": _maybe_stat(pop_df, "n_heads_mean", np.mean),
            "headcount_std_mean": _maybe_stat(pop_df, "n_heads_std", np.mean),
            "headcount_sum_mean": _maybe_stat(pop_df, "n_heads_sum", np.mean),
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("generation").reset_index(drop=True)
    return df


def plot_generation_lines(df: pd.DataFrame, metrics: Iterable[str], output_path: str) -> None:
    if df.empty:
        raise ValueError("No data to plot. Ensure checkpoints exist for the requested range.")

    metrics = list(metrics)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠️ Metric '{metric}' not found in aggregated DataFrame; skipping.")
            continue
        ax.plot(df["generation"], df[metric], marker="o", label=metric)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Metric Value")
    ax.set_title("NSGA-II Population Metrics by Generation")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_metrics_arg(raw: str) -> List[str]:
    if not raw:
        return DEFAULT_METRICS
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-generation metrics from NSGA-II checkpoints")
    parser.add_argument("--ckpt_base", type=str, default="ckpts/infi_medium_random/ckpt_gen", help="Base path to generation checkpoints (exclude generation number)")
    parser.add_argument("--start_gen", type=int, default=1, help="Starting generation index")
    parser.add_argument("--end_gen", type=int, default=50, help="Ending generation index")
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS), help="Comma-separated list of metrics to plot")
    parser.add_argument("--output", type=str, default="plots/generation_metrics.png", help="Output image file path")
    parser.add_argument("--csv", type=str, default="logs/generation_metrics.csv", help="Optional CSV path to save aggregated metrics")
    args = parser.parse_args()

    metrics_to_plot = parse_metrics_arg(args.metrics)

    df = compute_generation_metrics(args.ckpt_base, args.start_gen, args.end_gen)
    if df.empty:
        raise SystemExit("No metrics computed - check checkpoint paths and generation range.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"💾 Saved aggregated metrics to {args.csv}")

    plot_generation_lines(df, metrics_to_plot, args.output)
    print(f"✅ Saved generation metrics plot to {args.output}")


if __name__ == "__main__":
    main()
