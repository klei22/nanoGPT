"""Evaluate SmolLM2 before and after final-residual fine-tuning."""

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from .constants import DEFAULT_MODEL
from .model import SmolLM2FinalAttentionResidual

DEFAULT_TASKS = ("ifeval", "gsm8k", "minerva_math")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", default="out/smollm2-135m-final-attention-residual/benchmarks")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=float, help="Smoke-test on a fraction/count of each task")
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_harness_model(args, model_factory: Callable[..., Any] | None = None):
    # Pass model/tokenizer identifiers rather than initialized objects. Older
    # lm-eval releases forward non-string objects to Hugging Face auto loaders,
    # which fail with "spec must be str or dict".
    if model_factory is None:
        from lm_eval.models.huggingface import HFLM

        model_factory = HFLM
    return model_factory(
        pretrained=args.model,
        tokenizer=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )


def run_eval(args, adapter_path: str | None = None) -> dict:
    from lm_eval import evaluator

    harness_model = build_harness_model(args)
    if adapter_path is not None:
        adapter = SmolLM2FinalAttentionResidual(harness_model.model)
        adapter.load(adapter_path, map_location=args.device)
        harness_model.model.eval()

    results = evaluator.simple_evaluate(
        model=harness_model,
        tasks=args.tasks,
        limit=args.limit,
        log_samples=False,
        apply_chat_template=args.chat_template,
    )
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no results")
    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = run_eval(args)
    (output_dir / "before.json").write_text(json.dumps(before, indent=2, default=str))

    after = run_eval(args, adapter_path=args.adapter)
    (output_dir / "after.json").write_text(json.dumps(after, indent=2, default=str))

    comparison = {"before": before.get("results", {}), "after": after.get("results", {})}
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
