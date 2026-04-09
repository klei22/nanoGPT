#!/usr/bin/env python3
"""Evaluate a Hugging Face causal LM on HellaSwag using the repo's scoring logic."""
import argparse
import json
import math
import os
import shutil
from contextlib import nullcontext
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from datasets import config as datasets_config
from datasets.config import HF_DATASETS_CACHE, HF_MODULES_CACHE
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run HellaSwag evaluation with Hugging Face models")
	parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model id or local path")
	parser.add_argument(
		"--benchmark",
		type=str,
		default="hellaswag",
		choices=["hellaswag", "arc-easy", "arc-challenge", "sciq", "piqa", "winogrande", "boolq"],
		help="Dataset to evaluate (single)",
	)
	parser.add_argument(
		"--benchmarks",
		type=str,
		default=None,
		help="Comma-separated list of benchmarks to evaluate in one run",
	)
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
	parser.add_argument("--datasets_cache_dir", type=str, default=None, help="Optional datasets cache directory")
	parser.add_argument("--modules_cache_dir", type=str, default=None, help="Optional modules cache directory")
	return parser.parse_args()


def _parse_benchmarks(args: argparse.Namespace) -> List[str]:
	if args.benchmarks:
		items = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
		if not items:
			raise ValueError("--benchmarks must include at least one name")
		return items
	return [args.benchmark]


def _build_hellaswag_context(example: dict) -> str:
	ctx = example.get("ctx")
	if ctx:
		return ctx.strip()
	ctx_a = example.get("ctx_a", "").strip()
	ctx_b = example.get("ctx_b", "").strip()
	return (ctx_a + " " + ctx_b).strip()


def _get_benchmark_dataset(benchmark: str) -> tuple[str, str | None]:
	if benchmark == "arc-easy":
		return "ai2_arc", "ARC-Easy"
	if benchmark == "arc-challenge":
		return "ai2_arc", "ARC-Challenge"
	if benchmark == "sciq":
		return "sciq", None
	if benchmark == "piqa":
		return "piqa", None
	if benchmark == "winogrande":
		return "winogrande", "winogrande_xl"
	if benchmark == "boolq":
		return "boolq", None
	return "hellaswag", None



def _clear_modules_cache(modules_cache_dir: str, dataset_name: str) -> None:
	base = Path(modules_cache_dir) / "datasets"
	if not base.exists():
		return
	# Remove any cached dataset script folders/files matching this dataset.
	for path in base.rglob("*"):
		parts = set(path.parts)
		if dataset_name in parts:
			shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


def _load_dataset_with_retry(
	dataset_name: str,
	dataset_config: str | None,
	split: str,
	datasets_cache_dir: str | None,
	modules_cache_dir: str | None,
):
	try:
		if dataset_config is None:
			return load_dataset(dataset_name, split=split, cache_dir=datasets_cache_dir)
		return load_dataset(dataset_name, dataset_config, split=split, cache_dir=datasets_cache_dir)
	except UnicodeDecodeError:
		# Cache can get corrupted; clear module cache and force re-download.
		modules_dir = modules_cache_dir or datasets_config.HF_MODULES_CACHE or HF_MODULES_CACHE
		_clear_modules_cache(modules_dir, dataset_name)
		if dataset_config is None:
			return load_dataset(
				dataset_name,
				split=split,
				download_mode="force_redownload",
				cache_dir=datasets_cache_dir,
			)
		return load_dataset(
			dataset_name,
			dataset_config,
			split=split,
			download_mode="force_redownload",
			cache_dir=datasets_cache_dir,
		)


def _extract_example(benchmark: str, example: dict) -> tuple[str, List[str], int | None]:
	if benchmark == "hellaswag":
		ctx_text = _build_hellaswag_context(example)
		endings = example["endings"]
		label = example.get("label")
		return ctx_text, endings, label

	if benchmark in ("arc-easy", "arc-challenge"):
		ctx_text = example.get("question", "").strip()
		choices = example.get("choices", {})
		endings = choices.get("text", [])
		labels = choices.get("label", [])
		answer_key = example.get("answerKey")
		label = None
		if answer_key in labels:
			label = labels.index(answer_key)
		return ctx_text, endings, label

	if benchmark == "sciq":
		ctx_text = example.get("question", "").strip()
		endings = [
			example.get("correct_answer", ""),
			example.get("distractor1", ""),
			example.get("distractor2", ""),
			example.get("distractor3", ""),
		]
		return ctx_text, endings, 0

	if benchmark == "piqa":
		ctx_text = example.get("goal", "").strip()
		endings = [example.get("sol1", ""), example.get("sol2", "")]
		label = example.get("label")
		return ctx_text, endings, label

	if benchmark == "winogrande":
		sentence = example.get("sentence", "").strip()
		option1 = example.get("option1", "")
		option2 = example.get("option2", "")
		if "_" in sentence:
			before, after = sentence.split("_", 1)
			ctx_text = before
			endings = [option1 + after, option2 + after]
		else:
			ctx_text = sentence
			endings = [" " + option1, " " + option2]
		label_raw = example.get("answer")
		label = int(label_raw) - 1 if label_raw is not None else None
		return ctx_text, endings, label

	if benchmark == "boolq":
		passage = example.get("passage", "").strip()
		question = example.get("question", "").strip()
		ctx_text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
		endings = [" yes", " no"]
		answer = example.get("answer")
		label = 0 if answer is True else 1 if answer is False else None
		return ctx_text, endings, label

	raise ValueError(f"Unsupported benchmark: {benchmark}")


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

	if args.datasets_cache_dir:
		os.environ["HF_DATASETS_CACHE"] = args.datasets_cache_dir
		datasets_config.HF_DATASETS_CACHE = args.datasets_cache_dir
	if args.modules_cache_dir:
		os.environ["HF_MODULES_CACHE"] = args.modules_cache_dir
		datasets_config.HF_MODULES_CACHE = args.modules_cache_dir

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

	benchmarks = _parse_benchmarks(args)
	all_metrics = []

	for benchmark in benchmarks:
		dataset_name, dataset_config = _get_benchmark_dataset(benchmark)
		dataset = _load_dataset_with_retry(
			dataset_name,
			dataset_config,
			args.split,
			args.datasets_cache_dir,
			args.modules_cache_dir,
		)
		if args.max_examples is not None:
			dataset = dataset.shuffle(seed=args.seed).select(range(args.max_examples))

		correct = 0
		total = 0
		skipped = 0

		with torch.inference_mode():
			for example in dataset:
				ctx_text, endings, label = _extract_example(benchmark, example)
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
			"benchmark": benchmark,
		}
		all_metrics.append(metrics)
		print(json.dumps(metrics, indent=2))

	if args.output_json:
		with open(args.output_json, "w", encoding="utf-8") as f:
			json.dump(all_metrics, f, indent=2)
			f.write("\n")


if __name__ == "__main__":
	main()
