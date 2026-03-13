#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def run(cmd, log_path=None):
    print("+", " ".join(cmd))
    if log_path:
        with open(log_path, "w", encoding="utf-8") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def parse_ms_per_iter(log_path):
    ms_vals = []
    pat = re.compile(r",\s*([0-9]+\.?[0-9]*)\s*ms")
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                ms_vals.append(float(m.group(1)))
    if not ms_vals:
        return None
    return statistics.mean(ms_vals[-20:])


def load_param_count(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # take first row's num_params if present
    row = rows[0] if rows else {}
    for key in ("num_params", "params", "total_params"):
        if key in row and row[key]:
            return float(row[key])
    return None


def eval_loss(out_dir, dataset, eval_dir):
    run([
        "python3", "sample.py",
        "--out_dir", out_dir,
        "--eval_only",
        "--eval_dataset", dataset,
        "--eval_iters", "100",
        "--eval_output_dir", eval_dir,
    ])
    with open(os.path.join(eval_dir, "eval_loss.txt"), encoding="utf-8") as fh:
        data = json.load(fh)
    return float(data["val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shakespeare_char", choices=["shakespeare_char", "minipile"])
    ap.add_argument("--max-iters", type=int, default=1200)
    ap.add_argument("--output-root", default="out_abs_pos_ptq_demo")
    args = ap.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    variants = {
        "abs_default": ["--use_abs_pos_embeddings", "--abs_pos_embedding_variant", "default", "--no-use_rotary_embeddings"],
        "rope": ["--no-use_abs_pos_embeddings", "--use_rotary_embeddings"],
        "abs_cyclic": ["--use_abs_pos_embeddings", "--abs_pos_embedding_variant", "cyclic", "--abs_pos_cyclic_periods", "31", "47", "97", "--abs_pos_cyclic_random_start"],
    }

    speeds, params, quant = {}, {}, {}
    bits = list(range(8, 2, -1))

    for name, flags in variants.items():
        out_dir = root / f"{name}_{args.dataset}"
        train_log = root / f"{name}_train.log"
        run([
            "python3", "train.py", "--dataset", args.dataset,
            "--out_dir", str(out_dir), "--n_layer", "4", "--n_head", "4", "--n_embd", "256",
            "--block_size", "128", "--batch_size", "64", "--max_iters", str(args.max_iters),
            "--eval_interval", "400", "--eval_iters", "100", "--log_interval", "20", *flags,
        ], log_path=train_log)

        speeds[name] = parse_ms_per_iter(train_log)

        stats_csv = root / f"{name}_model_stats.csv"
        run(["python3", "train.py", "--dataset", args.dataset, "--out_dir", str(out_dir), "--eval_only", "--compute_model_stats", "--print_model_stats_table", str(stats_csv)])
        params[name] = load_param_count(stats_csv)

        baseline = eval_loss(str(out_dir), args.dataset, str(root / f"{name}_eval_fp32"))
        losses = []
        for b in bits:
            q_out = root / f"{name}_{b}bit"
            run(["python3", "quantizations/ptq/fake_quantize_ckpt.py", str(out_dir), "--num_bits", str(b), "--out_dir", str(q_out)])
            q_loss = eval_loss(str(q_out), args.dataset, str(root / f"{name}_eval_{b}bit"))
            losses.append((b, q_loss - baseline))
        quant[name] = losses

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(list(speeds.keys()), [speeds[k] or 0 for k in speeds])
    axes[0].set_title("Per-iteration speed")
    axes[0].set_ylabel("ms / iter (mean of last ~20 logs)")

    axes[1].bar(list(params.keys()), [params[k] or 0 for k in params])
    axes[1].set_title("Parameter count")
    axes[1].set_ylabel("parameters")

    for name, losses in quant.items():
        xb = [x for x, _ in losses]
        yb = [y for _, y in losses]
        axes[2].plot(xb, yb, marker="o", label=name)
    axes[2].invert_xaxis()
    axes[2].set_title("Quantizability (val loss delta vs fp32)")
    axes[2].set_xlabel("PTQ bits")
    axes[2].set_ylabel("Δ validation loss")
    axes[2].legend()

    fig.tight_layout()
    out_chart = root / "absolute_position_ptq_comparison.png"
    fig.savefig(out_chart, dpi=150)

    summary = {"speeds_ms_per_iter": speeds, "params": params, "quant_loss_delta": quant}
    with open(root / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote chart: {out_chart}")


if __name__ == "__main__":
    main()
