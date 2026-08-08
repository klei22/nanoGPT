#!/usr/bin/env python3
"""Train a LoRA control run using the same data and Trainer settings as FAR."""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--dataset_name", default="flytech/python-codes-25k")
    parser.add_argument("--dataset_config")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--train_split", default="train[:95%]")
    parser.add_argument("--eval_split", default="train[95%:]")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,v_proj")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    dataset_args = {"path": args.dataset_name}
    if args.dataset_config:
        dataset_args["name"] = args.dataset_config
    train = load_dataset(**dataset_args, split=args.train_split)
    validation = load_dataset(**dataset_args, split=args.eval_split)

    def tokenize(batch):
        texts = [f"# Programming problem or code\n{value}\n" for value in batch[args.text_column]]
        return tokenizer(texts, truncation=True, max_length=args.max_length, padding="max_length")

    train = train.map(tokenize, batched=True, remove_columns=train.column_names)
    validation = validation.map(tokenize, batched=True, remove_columns=validation.column_names)
    training_args = TrainingArguments(
        output_dir=str(output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=0.0, seed=args.seed,
        logging_steps=25, eval_strategy="steps", eval_steps=100,
        save_strategy="no",
        bf16=torch.cuda.is_available(), report_to="none", remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=validation,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    started = time.time()
    result = trainer.train()
    evaluation = trainer.evaluate()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "method": "lora", "base_model": args.model_name, "seed": args.seed,
        "trainable_parameters": trainable, "total_parameters": total,
        "wall_time_seconds": time.time() - started,
        "train_metrics": result.metrics, "eval_metrics": evaluation,
        "arguments": vars(args),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
