"""Benchmark checkpoints against standard LM-Eval tasks.

This script mirrors nanochat's focus on reporting evaluation metrics by
providing a lightweight entry-point that loads a checkpoint from the
main ``nanoGPT`` training pipeline, wraps it with the existing
``NanoGPTLM`` adapter and executes a configurable suite of lm-evaluation
benchmarks.  Results are written to disk and summarised in a compact
Rich table to make it easy to compare runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import tiktoken
from rich.console import Console
from rich.table import Table

from benchmarks.gpt_lm_eval_wrapper import NanoGPTLM
from model import GPT, GPTConfig
from utils.tokenizer_utils import get_tokenizer_functions
from variations.model_variations import model_variation_dictionary


DEFAULT_TASKS = [
    "arc_challenge",
    "arc_easy",
    "gsm8k",
    "mmlu",
    "humaneval",
]


@dataclass
class ModelLoadResult:
    model: GPT
    checkpoint: Dict[str, object] | None
    dataset_name: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks on a nanoGPT checkpoint")
    parser.add_argument("--out_dir", default="out", help="Directory containing the training run")
    parser.add_argument("--ckpt_name", default="ckpt.pt", help="Checkpoint filename to load from --out_dir")
    parser.add_argument("--init_from", default="resume", help="'resume' or a GPT-2 variant from variations/model_variations.py")
    parser.add_argument("--device", default="cuda", help="Computation device (cpu, cuda, cuda:0, ...)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Evaluation dtype")
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma separated list of lm-eval tasks (default: %(default)s)",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for lm-eval")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation length for generative tasks")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature used when lm-eval needs generation")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling for generation based tasks")
    parser.add_argument(
        "--results_output",
        type=str,
        default=None,
        help="Optional explicit path to the json file that will store raw lm-eval results",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional CSV file to store the condensed summary table",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Optional explicit path to a meta.pkl to use for tokenisation",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory for saving lm-eval outputs (defaults to <out_dir>/benchmarks)",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> ModelLoadResult:
    if args.init_from == "resume":
        ckpt_path = Path(args.out_dir) / args.ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=args.device)
        checkpoint_model_args = dict(checkpoint["model_args"])
        checkpoint_model_args["dropout"] = 0.0
        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for key in list(state_dict.keys()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)
        dataset_name = None
        config_section = checkpoint.get("config")
        if isinstance(config_section, dict):
            dataset_name = config_section.get("dataset")
        return ModelLoadResult(model=model, checkpoint=checkpoint, dataset_name=dataset_name)

    if args.init_from.startswith("gpt2"):
        gptconf = GPTConfig()
        variation_dict = model_variation_dictionary[args.init_from]
        for key, value in variation_dict.items():
            setattr(gptconf, key, value)
        model = GPT.from_pretrained(gptconf, model_type=args.init_from)
        return ModelLoadResult(model=model, checkpoint=None, dataset_name=None)

    raise ValueError("init_from must be 'resume' or one of the GPT-2 variations")


def resolve_tokenizer(
    args: argparse.Namespace,
    load_result: ModelLoadResult,
) -> Tuple:
    if args.init_from.startswith("gpt2"):
        enc = tiktoken.get_encoding("gpt2")
        return (
            lambda s: enc.encode(s, allowed_special={""}),
            lambda tokens: enc.decode(tokens),
        )

    candidate_meta_paths: List[Path] = []
    if args.meta_path:
        candidate_meta_paths.append(Path(args.meta_path))
    candidate_meta_paths.append(Path(args.out_dir) / "meta.pkl")
    if load_result.dataset_name:
        candidate_meta_paths.append(Path("data") / load_result.dataset_name / "meta.pkl")

    for meta_path in candidate_meta_paths:
        if meta_path.exists():
            with meta_path.open("rb") as f:
                meta = pickle.load(f)
            return get_tokenizer_functions(meta)

    raise FileNotFoundError(
        "Could not find a meta.pkl for tokenisation. Provide --meta_path or ensure the training run saved meta.pkl."
    )


def build_summary_table(results: Dict[str, Dict[str, float]]) -> Tuple[Table, List[Tuple[str, str, float]], float]:
    rows: List[Tuple[str, str, float]] = []
    for task, metrics in sorted(results.items()):
        if not metrics:
            continue
        metric_name, metric_value = next(iter(metrics.items()))
        rows.append((task, metric_name, float(metric_value)))

    table = Table(title="lm-eval summary")
    table.add_column("Task")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for task, metric_name, metric_value in rows:
        table.add_row(task, metric_name, f"{metric_value:.4f}")

    mean_score = float("nan")
    if rows:
        mean_score = sum(value for _, _, value in rows) / len(rows)
        table.add_section()
        table.add_row("mean", "-", f"{mean_score:.4f}")

    return table, rows, mean_score


def maybe_write_csv(path: str | None, rows: Iterable[Tuple[str, str, float]], mean_score: float) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "metric", "value"])
        for task, metric, value in rows:
            writer.writerow([task, metric, f"{value:.6f}"])
        writer.writerow(["mean", "-", f"{mean_score:.6f}"])


def main() -> None:
    args = parse_args()

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    load_result = load_model(args)
    model = load_result.model
    model.eval()
    model.to(args.device)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    eval_dtype = dtype_map[args.dtype]
    try:
        param_dtype = next(model.parameters()).dtype
    except StopIteration:
        param_dtype = eval_dtype
    if param_dtype != eval_dtype:
        model.to(dtype=eval_dtype)

    encode, decode = resolve_tokenizer(args, load_result)

    wrapped_model = NanoGPTLM(
        model=model,
        tokenizer_encode=encode,
        tokenizer_decode=decode,
        eot_token_id=model.config.vocab_size - 1,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    results_dir = Path(args.results_dir) if args.results_dir else Path(args.out_dir) / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.results_output
    if json_path is None:
        json_path = results_dir / f"{timestamp}_lm_eval_results.json"
    else:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.print(f"Running lm-eval on: {', '.join(tasks)}")

    results = wrapped_model.evaluate_and_save(
        tasks=tasks,
        batch_size=args.eval_batch_size,
        out_dir=str(results_dir),
        timestamp=timestamp,
        results_output=str(json_path),
    )

    table, rows, mean_score = build_summary_table(results.get("results", {}))
    console.print(table)

    summary_payload = {
        "tasks": tasks,
        "mean_score": mean_score,
        "results": rows,
        "timestamp": timestamp,
        "checkpoint": {
            "out_dir": args.out_dir,
            "ckpt_name": args.ckpt_name,
            "init_from": args.init_from,
        },
        "json_path": str(json_path),
    }

    summary_file = results_dir / f"{timestamp}_lm_eval_summary.json"
    with summary_file.open("w") as f:
        json.dump(summary_payload, f, indent=2)
        f.write("\n")

    maybe_write_csv(args.summary_csv, rows, mean_score)

    console.print(f"Raw results saved to {json_path}")
    console.print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
