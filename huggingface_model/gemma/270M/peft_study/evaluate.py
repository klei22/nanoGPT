#!/usr/bin/env python3
"""Evaluate a base, LoRA, or attention-residual checkpoint with lm-eval."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_residual_wrapper(path: Path, base_model: str, dtype: torch.dtype):
    script = path.parent.parent / "finetune_attention_residuals.py"
    spec = importlib.util.spec_from_file_location("attention_residual_training", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    base = AutoModelForCausalLM.from_pretrained(base_model, attn_implementation="eager", torch_dtype=dtype)
    model = module.AttentionResidualWrapper(base)
    state = torch.load(path / "attention_residuals.pt", map_location="cpu", weights_only=True)
    model.queries.data.copy_(state["attention_residual_queries"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("base", "lora", "attention_residual"), required=True)
    parser.add_argument("--base_model", default="google/gemma-3-270m")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks", default="arc_easy,hellaswag,piqa,winogrande")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--limit", type=float)
    args = parser.parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.method == "base":
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    elif args.method == "lora":
        if not args.checkpoint:
            parser.error("--checkpoint is required for lora")
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, args.checkpoint)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for attention_residual")
        model = load_residual_wrapper(Path(args.checkpoint), args.base_model, dtype)
    evaluator_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    results = simple_evaluate(
        model=evaluator_model, tasks=[task.strip() for task in args.tasks.split(",")],
        batch_size=args.batch_size, limit=args.limit, log_samples=False,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
