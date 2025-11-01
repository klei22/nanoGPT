# huggingface_model/gemma/270M/train_from_scratch.py
"""Train a Gemma 270M model from scratch on the FineWeb internet corpus."""
import os
import sys
from pathlib import Path

# Prevent GPU OOM on some systems
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
from typing import Dict, List
from types import SimpleNamespace

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset, load_dataset_builder
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from variations.norm_variations import HyperSphereNorm, RMSNorm


class EmbeddingWithNorm(nn.Module):
    """Wrap an embedding layer with an optional normalization module."""

    def __init__(self, embedding: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.norm = norm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeddings = self.embedding(input_ids)
        return self.norm(embeddings)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.embedding.weight

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def num_embeddings(self) -> int:
        return self.embedding.num_embeddings


def _build_embedding_norm(norm_type: str, embedding_dim: int) -> nn.Module:
    if norm_type == "rmsnorm":
        config = SimpleNamespace(n_embd=embedding_dim)
        return RMSNorm(config)
    if norm_type == "hyperspherenorm":
        config = SimpleNamespace(
            n_embd=embedding_dim,
            hsnorm_gain=True,
            hsnorm_radius=None,
            hsnorm_scale=1.0,
            hsnorm_radius_learning=False,
        )
        return HyperSphereNorm(config)
    raise ValueError(f"Unsupported embedding norm: {norm_type}")


class SampleOutputCallback(TrainerCallback):
    """Periodically log a short generation from the current model."""

    def __init__(self, tokenizer: AutoTokenizer, prompt: str = "The internet is a vast") -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt

    def on_log(self, args, state, control, **kwargs):  # type: ignore[override]
        model = kwargs.get("model")
        if model and state.is_world_process_zero:
            print(f"\n--- Sample output at step {state.global_step} ---")
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
            )
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = decoded_output.replace(self.prompt, "", 1).strip()
            print(self.prompt)
            print(generated_text)
            print("---------------------------------------\n")


def _tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer):
    return tokenizer(examples["text"])


def _group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(args: argparse.Namespace) -> None:
    model_name = "google/gemma-3-270m"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.precision in {"fp16", "bf16"} and not torch.cuda.is_available():
        raise ValueError("Mixed precision training requires CUDA availability.")

    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The current CUDA device does not support bfloat16 training.")

    print("Building model configuration and initializing weights from scratch...")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.resize_token_embeddings(len(tokenizer))

    if args.embedding_norm != "none":
        print(f"Attaching {args.embedding_norm} after token embeddings...")
        base_embeddings = model.get_input_embeddings()
        norm_layer = _build_embedding_norm(args.embedding_norm, base_embeddings.embedding_dim)
        model.set_input_embeddings(EmbeddingWithNorm(base_embeddings, norm_layer))

    print("Loading FineWeb dataset...")
    subset = args.fineweb_subset or None
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb", subset, split=args.dataset_split)
    except ValueError as err:
        if "BuilderConfig" in str(err):
            builder = load_dataset_builder("HuggingFaceFW/fineweb")
            available = ", ".join(sorted(builder.builder_configs))
            raise ValueError(
                "Unknown FineWeb subset '{subset_name}'. Available configurations: {options}".format(
                    subset_name=args.fineweb_subset,
                    options=available,
                )
            ) from err
        raise

    print("Splitting dataset...")
    if args.eval_fraction > 0:
        dataset = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
        train_dataset: Dataset = dataset["train"]
        eval_dataset: Dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print("Tokenizing dataset...")
    tokenized_train = train_dataset.map(
        lambda examples: _tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            lambda examples: _tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    print("Grouping tokenized texts into blocks...")
    lm_train = tokenized_train.map(
        lambda examples: _group_texts(examples, args.block_size),
        batched=True,
    )

    lm_eval = None
    if tokenized_eval is not None:
        lm_eval = tokenized_eval.map(
            lambda examples: _group_texts(examples, args.block_size),
            batched=True,
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.total_iterations,
        logging_strategy="steps",
        eval_strategy="steps" if lm_eval is not None else "no",
        save_strategy="steps",
        logging_steps=args.sample_frequency,
        eval_steps=args.sample_frequency if lm_eval is not None else None,
        save_steps=args.sample_frequency,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        push_to_hub=False,
    )

    callbacks = [SampleOutputCallback(tokenizer=tokenizer, prompt=args.sample_prompt)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Starting training from scratch...")
    trainer.train()
    trainer.save_model()
    print("Training complete! Model saved to", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Gemma 270M model from scratch on FineWeb.")
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train[:1%]",
        help="Split of FineWeb to load (HF dataset slicing syntax).",
    )
    parser.add_argument(
        "--fineweb_subset",
        type=str,
        default="sample-10BT",
        help=(
            "Subset configuration of FineWeb to load. Use an empty string to select the default dataset "
            "configuration."
        ),
    )
    parser.add_argument(
        "--eval_fraction",
        type=float,
        default=0.01,
        help="Fraction of data reserved for evaluation (0 to disable).",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Number of tokens per training example after grouping.",
    )
    parser.add_argument(
        "--total_iterations",
        type=int,
        default=5000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1000,
        help="Steps between logging, saving, and evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=200,
        help="Number of warmup steps for the LR scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("none", "fp16", "bf16"),
        default="none",
        help=(
            "Mixed precision mode to use. Set to 'fp16' or 'bf16' to enable the corresponding"
            " autocast, or leave as 'none' for full float32 training."
        ),
    )
    parser.add_argument(
        "--embedding_norm",
        type=str,
        choices=("none", "rmsnorm", "hyperspherenorm"),
        default="none",
        help="Optional normalization layer to insert immediately after the token embeddings.",
    )
    parser.add_argument(
        "--sample_prompt",
        type=str,
        default="The internet is a vast",
        help="Prompt used for periodic generation samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemma-3-270m-fineweb-from-scratch",
        help="Directory to store checkpoints and final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
