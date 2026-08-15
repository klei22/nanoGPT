#!/usr/bin/env python3
"""Fine-tune a small Hugging Face LM's attention-residual adapter on GSM8K."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_model.attention_residual_peft.adapter import attach_attention_residual_peft


ANSWER_RE = re.compile(r"####\s*([-+]?[$\d,]*\.?\d+)")
NUMBER_RE = re.compile(r"[-+]?[$\d,]*\.?\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output_dir", default="out/smollm2-135m-gsm8k-attnres")
    parser.add_argument("--adapter_dir", default=None, help="Evaluate or continue a saved adapter")
    parser.add_argument("--train_examples", type=int, default=1000)
    parser.add_argument("--eval_examples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def prompt(question: str) -> str:
    return f"Solve the problem. Show your reasoning, then write the final answer after ####.\n\nQuestion: {question}\nAnswer:"


def normalized_answer(text: str, reference: bool = False) -> str | None:
    matches = ANSWER_RE.findall(text)
    if not matches and not reference:
        matches = NUMBER_RE.findall(text)
    return matches[-1].replace("$", "").replace(",", "") if matches else None


def evaluate(model, tokenizer, examples, args: argparse.Namespace) -> dict:
    correct = 0
    records = []
    model.eval()
    for example in examples:
        encoded = tokenizer(prompt(example["question"]), return_tensors="pt", truncation=True, max_length=args.max_length)
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(output[0, encoded["input_ids"].size(1):], skip_special_tokens=True)
        prediction = normalized_answer(completion)
        target = normalized_answer(example["answer"], reference=True)
        correct += prediction == target
        records.append({"question": example["question"], "target": target, "prediction": prediction, "completion": completion})
    return {"correct": correct, "total": len(records), "exact_match": correct / len(records) if records else 0.0, "records": records}


def batch_loss(model, tokenizer, examples, args: argparse.Namespace) -> torch.Tensor:
    texts = [prompt(item["question"]) + " " + item["answer"] + tokenizer.eos_token for item in examples]
    prefixes = [prompt(item["question"]) for item in examples]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    labels = batch["input_ids"].clone()
    for row, prefix in enumerate(prefixes):
        prefix_length = len(tokenizer(prefix, truncation=True, max_length=args.max_length)["input_ids"])
        labels[row, :prefix_length] = -100
    labels[batch["attention_mask"] == 0] = -100
    inputs = {key: value.to(args.device) for key, value in batch.items()}
    return model(**inputs, labels=labels.to(args.device)).loss


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and args.device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    adapter = attach_attention_residual_peft(model, args.adapter_dir)
    adapter_parameters = sum(p.numel() for p in adapter.parameters())
    base_parameters = sum(p.numel() for p in model.parameters())
    print(f"trainable adapter parameters: {adapter_parameters:,} ({100 * adapter_parameters / base_parameters:.4f}% of base)")

    dataset = load_dataset("openai/gsm8k", "main")
    train = list(dataset["train"].shuffle(seed=args.seed).select(range(min(args.train_examples, len(dataset["train"])))))
    test = list(dataset["test"].select(range(min(args.eval_examples, len(dataset["test"])))))
    before = evaluate(model, tokenizer, test, args)

    train_losses = []
    if not args.eval_only:
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for _ in range(args.epochs):
            random.shuffle(train)
            for offset in range(0, len(train), args.batch_size):
                loss = batch_loss(model, tokenizer, train[offset:offset + args.batch_size], args)
                train_losses.append(loss.item())
                (loss / args.gradient_accumulation).backward()
                step += 1
                if step % args.gradient_accumulation == 0 or offset + args.batch_size >= len(train):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % 10 == 0:
                    print(f"step {step}: loss={loss.item():.4f}")
        adapter.save(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    after = before if args.eval_only else evaluate(model, tokenizer, test, args)
    result = {
        "model": args.model,
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "adapter_percent": 100 * adapter_parameters / base_parameters,
        "train_losses": train_losses,
        "before": before,
        "after": after,
        "config": vars(args),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gsm8k_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"before": {k: v for k, v in before.items() if k != "records"}, "after": {k: v for k, v in after.items() if k != "records"}}, indent=2))


if __name__ == "__main__":
    main()
