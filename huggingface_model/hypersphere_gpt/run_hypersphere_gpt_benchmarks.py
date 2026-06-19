"""Run the Hypersphere GPT model on standard Hugging Face evaluation benchmarks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

try:
    import lm_eval
except ImportError as exc:  # pragma: no cover - optional dependency for runtime
    raise RuntimeError(
        "lm-evaluation-harness must be installed to run the benchmark script"
    ) from exc


DEFAULT_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or Hugging Face identifier of the exported Hypersphere GPT model",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="gpt2",
        help="Tokenizer identifier used for evaluation (default: gpt2)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma separated list of lm-eval tasks to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples to use for each task",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of evaluation samples per task",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Computation dtype to use in evaluation (e.g. float32, bfloat16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device specification passed to lm-eval (e.g. cuda, cpu)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional path to write the aggregated benchmark JSON results",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading custom code when using remote checkpoints",
    )
    return parser.parse_args()


def build_model_args(model_path: str, device: str, dtype: str, trust_remote_code: bool) -> str:
    args: List[str] = [f"pretrained={model_path}"]
    if device:
        args.append(f"device={device}")
    if dtype:
        args.append(f"dtype={dtype}")
    if trust_remote_code:
        args.append("trust_remote_code=True")
    return ",".join(args)


def main() -> None:
    args = parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one benchmark task must be provided")

    model_args = build_model_args(
        args.model_path, args.device, args.dtype, args.trust_remote_code
    )
    tokenizer_args = f"pretrained={args.tokenizer_name}"

    print(f"Running lm-eval tasks: {', '.join(tasks)}")
    results = lm_eval.simple_evaluate(
        model="hf-causal",
        tokenizer="hf",
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        tasks=tasks,
        batch_size=args.batch_size,
        num_fewshot=args.fewshot,
        limit=args.limit,
    )

    print(json.dumps(results["results"], indent=2))

    output_path = args.output_path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"lm_eval_results_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Saved full benchmark results to {output_path}")


if __name__ == "__main__":
    main()
