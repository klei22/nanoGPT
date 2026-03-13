#!/usr/bin/env python3
"""Evaluate a trained checkpoint on HellaSwag using repo tokenizer/config.

This follows the same checkpoint and tokenizer loading flow as sample.py.
"""
import argparse
import json
import math
import os
import pickle
import sys
from contextlib import nullcontext
from dataclasses import fields
from inspect import signature
from typing import List

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import GPT, GPTConfig
from sample import get_tokenizer_functions
from variations.model_variations import model_variation_dictionary
from benchmarks.evaluate_huggingface_models import (
    _extract_example,
    _get_benchmark_dataset,
    _get_block_size,
    _load_dataset_with_retry,
    _parse_benchmarks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HellaSwag evaluation")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory containing ckpt.pt and meta.pkl")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional explicit path to ckpt.pt")
    parser.add_argument("--config_path", type=str, default=None, help="Optional JSON config to supplement model_args")
    parser.add_argument("--init_from", type=str, default="resume", help="'resume' or a GPT-2 variant")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Autocast dtype")
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
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional cap on number of examples")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling")
    parser.add_argument("--block_size", type=int, default=None, help="Override model block size")
    parser.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True, help="Normalize by ending length")
    parser.add_argument("--weights_only", default=False, action=argparse.BooleanOptionalAction, help="Disable to allow full pickle loading for legacy checkpoints")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write metrics JSON")
    parser.add_argument("--datasets_cache_dir", type=str, default=None, help="Optional datasets cache directory")
    parser.add_argument("--modules_cache_dir", type=str, default=None, help="Optional modules cache directory")
    parser.add_argument(
        "--print_examples",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Print one correct and one incorrect example with option probabilities",
    )
    return parser.parse_args()


def _load_json_config(config_path: str) -> tuple[dict, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config JSON must be an object")

    if "model_args" in raw and isinstance(raw["model_args"], dict):
        return raw["model_args"], raw.get("config", raw)

    return raw, raw


def _load_checkpoint(args: argparse.Namespace):
    ckpt_path = args.ckpt_path or os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_kwargs = {"map_location": args.device}
    if "weights_only" in signature(torch.load).parameters:
        load_kwargs["weights_only"] = args.weights_only

    checkpoint = torch.load(ckpt_path, **load_kwargs)
    checkpoint_config = checkpoint.get("config", {})

    config_override = None
    config_meta_hint = None
    if args.config_path:
        config_override, config_meta_hint = _load_json_config(args.config_path)
        if not checkpoint_config:
            checkpoint_config = config_meta_hint

    if args.init_from == "resume":
        model_args = {}
        if isinstance(checkpoint.get("model_args"), dict):
            model_args.update(checkpoint["model_args"])
        if config_override:
            model_args.update(config_override)
        if not model_args:
            raise ValueError("No model_args found in checkpoint; provide --config_path.")

        allowed_keys = {field.name for field in fields(GPTConfig)}
        model_args = {k: v for k, v in model_args.items() if k in allowed_keys}

        model_args["dropout"] = 0.0
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    else:
        gptconf = GPTConfig()
        variation_dict = model_variation_dictionary[args.init_from]
        for k, v in variation_dict.items():
            setattr(gptconf, k, v)
        model = GPT.from_pretrained(gptconf, model_type=args.init_from)

    return model, checkpoint_config


def _load_tokenizer(args: argparse.Namespace, checkpoint_config: dict):
    meta_paths: List[str] = []
    meta_paths.append(os.path.join(args.out_dir, "meta.pkl"))

    dataset_name = checkpoint_config.get("dataset") if isinstance(checkpoint_config, dict) else None
    if dataset_name:
        meta_paths.append(os.path.join("data", dataset_name, "meta.pkl"))

    for meta_path in meta_paths:
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            encode, decode = get_tokenizer_functions(meta)
            return encode, decode

    if args.init_from.startswith("gpt2"):
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)
        return encode, decode

    raise FileNotFoundError("No meta.pkl found and no GPT-2 fallback available.")


def _score_example(
    model: GPT,
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

        if len(end_tokens) > block_size:
            end_tokens = end_tokens[-block_size:]

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
            logits, _ = model(input_ids, target_ids)
        logprobs = torch.log_softmax(logits, dim=-1)
        target_slice = target_ids[:, ending_start:]
        lp = logprobs[:, ending_start:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)

        if length_norm:
            score = lp.mean().item()
        else:
            score = lp.sum().item()
        scores.append(score)

    return scores


def _score_example_with_tokens(
    model: GPT,
    encode,
    decode,
    ctx_text: str,
    endings: List[str],
    block_size: int,
    length_norm: bool,
    device: str,
    ctx_autocast,
) -> tuple[List[float], List[List[str]], List[List[float]]]:
    ctx_tokens = encode(ctx_text)
    scores: List[float] = []
    token_texts: List[List[str]] = []
    token_logprobs: List[List[float]] = []

    for ending in endings:
        end_tokens = encode(ending)
        if len(end_tokens) == 0:
            scores.append(-math.inf)
            token_texts.append([])
            token_logprobs.append([])
            continue

        if len(end_tokens) > block_size:
            end_tokens = end_tokens[-block_size:]

        max_ctx_len = max(0, block_size - len(end_tokens))
        if len(ctx_tokens) > max_ctx_len:
            ctx_trim = ctx_tokens[-max_ctx_len:]
        else:
            ctx_trim = ctx_tokens

        full = ctx_trim + end_tokens
        if len(full) < 2:
            scores.append(-math.inf)
            token_texts.append([])
            token_logprobs.append([])
            continue

        input_ids = torch.tensor(full[:-1], device=device).unsqueeze(0)
        target_ids = torch.tensor(full[1:], device=device).unsqueeze(0)
        ending_start = max(len(ctx_trim) - 1, 0)

        with ctx_autocast:
            logits, _ = model(input_ids, target_ids)
        logprobs = torch.log_softmax(logits, dim=-1)
        target_slice = target_ids[:, ending_start:]
        lp = logprobs[:, ending_start:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)

        if length_norm:
            score = lp.mean().item()
        else:
            score = lp.sum().item()
        scores.append(score)

        end_token_ids = target_slice[0].tolist()
        token_texts.append([decode([t]) for t in end_token_ids])
        token_logprobs.append(lp[0].tolist())

    return scores, token_texts, token_logprobs


def _print_example(
    tag: str,
    ctx_text: str,
    endings: List[str],
    probs: List[float],
    label: int,
    pred: int,
    token_texts: List[List[str]] | None = None,
    token_logprobs: List[List[float]] | None = None,
) -> None:
    print("\n" + "=" * 80)
    print(f"{tag}: predicted={pred} label={label}")
    print("Context:")
    print(ctx_text)
    print("Options:")
    for i, (ending, prob) in enumerate(zip(endings, probs)):
        print(f"[{i}] p={prob:.4f} {ending}")
        if token_texts is not None and token_logprobs is not None:
            if i < len(token_texts) and i < len(token_logprobs):
                print("  token logprobs:")
                for token, lp in zip(token_texts[i], token_logprobs[i]):
                    print(f"    {token!r}: {lp:.4f}")
                if token_logprobs[i]:
                    avg_lp = sum(token_logprobs[i]) / len(token_logprobs[i])
                    print(f"  avg token logprob: {avg_lp:.4f}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()

    if args.datasets_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.datasets_cache_dir
    if args.modules_cache_dir:
        os.environ["HF_MODULES_CACHE"] = args.modules_cache_dir

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, checkpoint_config = _load_checkpoint(args)
    encode, decode = _load_tokenizer(args, checkpoint_config)

    model.eval()
    model.to(args.device)

    block_size = int(getattr(model.config, "block_size", _get_block_size(model, args.block_size)))
    if args.block_size is not None and args.block_size != block_size:
        print(f"[info] Ignoring --block_size={args.block_size}; using checkpoint block_size={block_size}.")
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    ctx_autocast = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
        printed_correct = False
        printed_incorrect = False
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
                    if args.print_examples and not printed_correct:
                        scores_dbg, token_texts, token_logprobs = _score_example_with_tokens(
                            model=model,
                            encode=encode,
                            decode=decode,
                            ctx_text=ctx_text,
                            endings=endings,
                            block_size=block_size,
                            length_norm=args.length_norm,
                            device=args.device,
                            ctx_autocast=ctx_autocast,
                        )
                        probs = torch.softmax(torch.tensor(scores_dbg), dim=-1).tolist()
                        _print_example(
                            f"CORRECT ({benchmark})",
                            ctx_text,
                            endings,
                            probs,
                            label,
                            pred,
                            token_texts=token_texts,
                            token_logprobs=token_logprobs,
                        )
                        printed_correct = True
                total += 1
                if pred != int(label) and args.print_examples and not printed_incorrect:
                    scores_dbg, token_texts, token_logprobs = _score_example_with_tokens(
                        model=model,
                        encode=encode,
                        decode=decode,
                        ctx_text=ctx_text,
                        endings=endings,
                        block_size=block_size,
                        length_norm=args.length_norm,
                        device=args.device,
                        ctx_autocast=ctx_autocast,
                    )
                    probs = torch.softmax(torch.tensor(scores_dbg), dim=-1).tolist()
                    _print_example(
                        f"INCORRECT ({benchmark})",
                        ctx_text,
                        endings,
                        probs,
                        label,
                        pred,
                        token_texts=token_texts,
                        token_logprobs=token_logprobs,
                    )
                    printed_incorrect = True

        acc = (correct / total) if total else float("nan")
        metrics = {
            "split": args.split,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "skipped": skipped,
            "block_size": block_size,
            "length_norm": bool(args.length_norm),
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
