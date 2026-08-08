#!/usr/bin/env python3
"""Fine-tune Gemma 3 270M with only new attention-residual parameters trainable.

This mirrors nanoGPT's FullAttentionResidual idea from ``model.py``: each layer
can receive a learned depth-wise mix of the embedding stream and earlier layer
outputs.  The base Hugging Face model is frozen; optimization updates only the
new zero-initialized query vectors in ``AttentionResidualWrapper``.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


@dataclass
class ResidualState:
    """Per-forward-pass storage shared by hooks."""

    sources: list[torch.Tensor]
    destination: int = 0


class AttentionResidualWrapper(nn.Module):
    """Add nanoGPT-style full attention residuals to a decoder-only HF model.

    ``queries`` are the only trainable parameters.  A forward pre-hook replaces
    each decoder layer input with a token-local softmax mix over previous hidden
    states.  A forward hook records the layer output as the next source.  This
    provides a lightweight way to fine-tune residual routing while all original
    model weights remain frozen.
    """

    def __init__(self, base_model: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.base_model = base_model
        self.eps = eps
        self.layers = self._find_decoder_layers(base_model)
        hidden_size = int(base_model.config.hidden_size)
        self.queries = nn.Parameter(torch.zeros(len(self.layers), hidden_size))
        self._state: ResidualState | None = None
        self._handles = []
        self._install_hooks()
        self._freeze_base_model()

    @staticmethod
    def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
        """Find the canonical HF decoder layer stack (Gemma: model.model.layers)."""
        candidate_paths = ("model.layers", "language_model.model.layers", "transformer.h", "gpt_neox.layers")
        for path in candidate_paths:
            module: Any = model
            for part in path.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return module
        raise ValueError("Could not locate decoder layers; add the layer path for this model architecture.")

    def _freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.queries.requires_grad = True

    @property
    def config(self):
        """Expose the wrapped configuration to Hugging Face evaluators."""
        return self.base_model.config

    @property
    def device(self):
        """Expose the wrapped model device to Hugging Face evaluators."""
        return self.queries.device

    def _install_hooks(self) -> None:
        for layer_index, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_pre_hook(self._make_pre_hook(layer_index), with_kwargs=True))
            self._handles.append(layer.register_forward_hook(self._make_post_hook(), with_kwargs=True))

    def _make_pre_hook(self, layer_index: int):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            if self._state is None or not args:
                return args, kwargs
            mixed = self._mix_sources(self._state.sources, layer_index)
            return (mixed, *args[1:]), kwargs

        return hook

    def _make_post_hook(self):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any):
            if self._state is None:
                return output
            hidden_states = output[0] if isinstance(output, tuple) else output
            self._state.sources.append(hidden_states)
            self._state.destination += 1
            return output

        return hook

    def _mix_sources(self, sources: list[torch.Tensor], destination: int) -> torch.Tensor:
        values = torch.stack(sources, dim=0)  # depth, batch, time, channels
        keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, self.queries[destination].to(keys.dtype))
        weights = scores.softmax(dim=0)
        return torch.einsum("dbt,dbtc->btc", weights, values)

    def forward(self, *args: Any, **kwargs: Any):
        input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
        if input_ids is None:
            raise ValueError("AttentionResidualWrapper expects input_ids so it can seed residual sources.")
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        self._state = ResidualState(sources=[embeddings])
        try:
            return self.base_model(*args, **kwargs)
        finally:
            self._state = None

    def save_trainable_parameters(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save({"attention_residual_queries": self.queries.detach().cpu()}, os.path.join(output_dir, "attention_residuals.pt"))


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def format_example(example: dict[str, Any], text_column: str) -> str:
    text = example[text_column]
    return f"# Programming problem or code\n{text}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only Gemma attention-residual query parameters on code data.")
    parser.add_argument("--model_name", default="google/gemma-3-270m")
    parser.add_argument("--dataset_name", default="flytech/python-codes-25k")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--train_split", default="train[:95%]")
    parser.add_argument("--eval_split", default="train[95%:]")
    parser.add_argument("--output_dir", default="./gemma-3-270m-attention-residual-code")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--attention_residual_eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = AttentionResidualWrapper(base_model, eps=args.attention_residual_eps)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.6f}%)")

    dataset_kwargs = {"path": args.dataset_name}
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    train_dataset = load_dataset(**dataset_kwargs, split=args.train_split)
    eval_dataset = load_dataset(**dataset_kwargs, split=args.eval_split)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [format_example({args.text_column: value}, args.text_column) for value in batch[args.text_column]]
        return tokenizer(texts, truncation=True, max_length=args.max_length, padding="max_length")

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # Saving the wrapper state would duplicate the frozen base checkpoint.
        # Persist only the residual tensor explicitly after training instead.
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    started = time.time()
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    model.save_trainable_parameters(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "method": "attention_residual",
        "base_model": args.model_name,
        "seed": args.seed,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "wall_time_seconds": time.time() - started,
        "train_metrics": train_result.metrics,
        "eval_metrics": evaluation,
        "arguments": vars(args),
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
