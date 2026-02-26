#!/usr/bin/env python3
"""Evaluate a Hugging Face causal LM on HellaSwag using the repo's scoring logic."""
import argparse
import json
import math
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run HellaSwag evaluation with Hugging Face models")
	parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model id or local path")
	parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation")
	parser.add_argument(
		"--dtype",
		type=str,
		default="bfloat16",
		choices=["bfloat16", "float16", "float32"],
		help="Autocast dtype",
	)
	parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
	parser.add_argument("--max_examples", type=int, default=None, help="Optional cap on number of examples")
	parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling")
	parser.add_argument("--block_size", type=int, default=1024, help="Override model block size")
	parser.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--output_json", type=str, default=None, help="Optional path to write metrics JSON")
	return parser.parse_args()


def _build_context(example: dict) -> str:
	ctx = example.get("ctx")
	if ctx:
		return ctx.strip()
	ctx_a = example.get("ctx_a", "").strip()
	ctx_b = example.get("ctx_b", "").strip()
	return (ctx_a + " " + ctx_b).strip()


def _get_block_size(model, override: Optional[int]) -> int:
	if override is not None:
		return override
	if hasattr(model.config, "n_positions") and model.config.n_positions:
		return int(model.config.n_positions)
	if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
		return int(model.config.max_position_embeddings)
	return 1024


def _score_example(
	model,
	encode,
	ctx_text: str,
	endings: List[str],
	block_size: int,
	length_norm: bool,
	device: str,
	ctx_autocast,
) -> List[float]:
	ctx_tokens = encode(ctx_text)
	scores: List[float] = []

	for ending in endings:
		end_tokens = encode(ending)
		if len(end_tokens) == 0:
			scores.append(-math.inf)
			continue

		max_ctx_len = max(0, block_size - len(end_tokens))
		if len(ctx_tokens) > max_ctx_len:
			ctx_trim = ctx_tokens[-max_ctx_len:]
		else:
			ctx_trim = ctx_tokens

		full = ctx_trim + end_tokens
		if len(full) < 2:
			scores.append(-math.inf)
			continue

		input_ids = torch.tensor(full[:-1], device=device).unsqueeze(0)
		target_ids = torch.tensor(full[1:], device=device).unsqueeze(0)
		ending_start = max(len(ctx_trim) - 1, 0)

		with ctx_autocast:
			logits = model(input_ids).logits
		logprobs = torch.log_softmax(logits, dim=-1)
		target_slice = target_ids[:, ending_start:]
		lp = logprobs[:, ending_start:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)

		if length_norm:
			score = lp.mean().item()
		else:
			score = lp.sum().item()
		scores.append(score)

	return scores


def main() -> None:
	args = parse_args()

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
	if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(args.model_name)
	model.eval()
	model.to(args.device)

	block_size = _get_block_size(model, args.block_size)
	device_type = "cuda" if "cuda" in args.device else "cpu"
	ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
	ctx_autocast = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

	encode = lambda s: tokenizer.encode(s, add_special_tokens=False)

	dataset = load_dataset("hellaswag", split=args.split)
	if args.max_examples is not None:
		dataset = dataset.shuffle(seed=args.seed).select(range(args.max_examples))

	correct = 0
	total = 0
	skipped = 0

	with torch.inference_mode():
		for example in dataset:
			ctx_text = _build_context(example)
			endings = example["endings"]
			label = example.get("label")
			if label is None:
				skipped += 1
				continue

			scores = _score_example(
				model=model,
				encode=encode,
				ctx_text=ctx_text,
				endings=endings,
				block_size=block_size,
				length_norm=args.length_norm,
				device=args.device,
				ctx_autocast=ctx_autocast,
			)

			pred = int(np.argmax(scores))
			if pred == int(label):
				correct += 1
			total += 1

	acc = (correct / total) if total else float("nan")
	metrics = {
		"split": args.split,
		"total": total,
		"correct": correct,
		"accuracy": acc,
		"skipped": skipped,
		"block_size": block_size,
		"length_norm": bool(args.length_norm),
		"model_name": args.model_name,
	}

	print(json.dumps(metrics, indent=2))

	if args.output_json:
		with open(args.output_json, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)
			f.write("\n")


if __name__ == "__main__":
	main()
