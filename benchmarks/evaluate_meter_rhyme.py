#!/usr/bin/env python3
"""Evaluate meter/rhyme JSONL with conditional likelihood and cluster CIs."""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def macro_f1(gold, predicted):
    labels = set(gold) | set(predicted)
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def normalized_edit_distance(left, right):
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def cluster_interval(rows, metric, samples=1000, seed=1337):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    keys = sorted(groups)
    if not keys:
        return [0.0, 0.0]
    rng, values = random.Random(seed), []
    for _ in range(samples):
        sampled = [item for _ in keys for item in groups[rng.choice(keys)]]
        values.append(metric(sampled))
    values.sort()
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def summarize(rows):
    accuracy = lambda subset: sum(r["correct"] for r in subset) / len(subset) if subset else 0.0
    results = {"examples": len(rows), "accuracy": accuracy(rows),
               "accuracy_cluster_95ci": cluster_interval(rows, accuracy)}
    margins = [r["margin"] for r in rows if r.get("margin") is not None]
    if margins:
        results["mean_likelihood_margin"] = sum(margins) / len(margins)
    meter = [r for r in rows if r.get("gold_meter") is not None]
    if meter:
        results["meter_family_macro_f1"] = macro_f1([r["gold_meter"] for r in meter], [r["pred_meter"] for r in meter])
    scansion = [r for r in rows if r.get("gold_scansion") is not None]
    if scansion:
        results["exact_scansion_accuracy"] = sum(r["gold_scansion"] == r["pred_scansion"] for r in scansion) / len(scansion)
        results["scansion_normalized_edit_distance"] = sum(normalized_edit_distance(r["gold_scansion"], r["pred_scansion"]) for r in scansion) / len(scansion)
    rhyme = [r for r in rows if r.get("gold_rhyme") is not None]
    if rhyme:
        results["exact_rhyme_scheme_accuracy"] = sum(r["gold_rhyme"] == r["pred_rhyme"] for r in rhyme) / len(rhyme)
    joint = [r for r in rows if r.get("meter_success") is not None and r.get("rhyme_success") is not None]
    if joint:
        results["joint_success_rate"] = sum(r["meter_success"] and r["rhyme_success"] for r in joint) / len(joint)
    results["per_source"] = {source: {"examples": len(items), "accuracy": accuracy(items)}
                             for source, items in sorted(_group(rows, "source").items())}
    results["per_task"] = {task: {"examples": len(items), "accuracy": accuracy(items)}
                           for task, items in sorted(_group(rows, "task").items())}
    return results


def _group(rows, key):
    output = defaultdict(list)
    for row in rows:
        output[row.get(key, "unknown")].append(row)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--tasks", default=None, help="Comma-separated task filter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json")
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--config_path")
    parser.add_argument("--init_from", default="resume")
    parser.add_argument("--weights_only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    import torch
    from benchmarks.evaluate_custom_models import _load_checkpoint, _load_tokenizer, _score_example

    args = parse_args()
    rows = [json.loads(line) for line in open(args.benchmark_file, encoding="utf-8") if line.strip()]
    if args.tasks:
        wanted = set(args.tasks.split(","))
        rows = [row for row in rows if row["task"] in wanted]
    model, config = _load_checkpoint(args)
    model.eval().to(args.device)
    encode, _ = _load_tokenizer(args, config)
    block_size = args.block_size or getattr(model.config, "block_size", 1024)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=dtype)
    evaluated = []
    for row in rows:
        if len(row["choices"]) < 1:
            continue
        scores = _score_example(model, encode, row["context"], row["choices"], block_size,
                                args.length_norm, args.device, autocast)
        prediction = max(range(len(scores)), key=scores.__getitem__)
        result = {"id": row["id"], "group_id": row["group_id"], "task": row["task"],
                  "source": row.get("source", "unknown"), "prediction": prediction,
                  "label": row["label"], "correct": prediction == row["label"], "scores": scores,
                  "margin": scores[row["label"]] - max((s for i, s in enumerate(scores) if i != row["label"]), default=scores[row["label"]])}
        for name in ("meter", "scansion", "rhyme"):
            values = row.get(f"choice_{name}s")
            if values:
                result[f"gold_{name}"] = values[row["label"]]
                result[f"pred_{name}"] = values[prediction]
        result["meter_success"] = row.get("choice_meter_success", [None] * len(scores))[prediction]
        result["rhyme_success"] = row.get("choice_rhyme_success", [None] * len(scores))[prediction]
        evaluated.append(result)
    output = {"benchmark_file": os.path.abspath(args.benchmark_file), "metrics": summarize(evaluated), "results": evaluated}
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
