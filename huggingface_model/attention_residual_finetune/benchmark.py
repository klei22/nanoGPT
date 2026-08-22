"""Evaluate SmolLM2 before and after final-residual fine-tuning."""

import argparse
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import SmolLM2FinalAttentionResidual
from .train import DEFAULT_MODEL

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


def run_eval(model, tokenizer, args) -> dict:
    harness_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    return evaluator.simple_evaluate(
        model=harness_model,
        tasks=args.tasks,
        limit=args.limit,
        log_samples=False,
        apply_chat_template=args.chat_template,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    before = run_eval(model, tokenizer, args)
    (output_dir / "before.json").write_text(json.dumps(before, indent=2, default=str))

    adapter = SmolLM2FinalAttentionResidual(model)
    adapter.load(args.adapter)
    model.eval()
    after = run_eval(model, tokenizer, args)
    (output_dir / "after.json").write_text(json.dumps(after, indent=2, default=str))

    comparison = {"before": before.get("results", {}), "after": after.get("results", {})}
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
