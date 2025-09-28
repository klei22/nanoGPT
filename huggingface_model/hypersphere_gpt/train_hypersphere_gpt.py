import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from .config import HypersphereGPTConfig
from .modeling_hypersphere_gpt import HypersphereGPTForCausalLM
from .tokenizer import TiktokenTokenizer


@dataclass
class ConstantLengthDataset(Dataset):
    examples: List[List[int]]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        ids = self.examples[idx]
        return {"input_ids": ids, "labels": ids}


def tokenize_example(tokenizer: TiktokenTokenizer, text: str) -> List[int]:
    return tokenizer.encode(text)


def build_dataset(tokenizer: TiktokenTokenizer, block_size: int, dataset_name: str, dataset_split: str) -> ConstantLengthDataset:
    raw_dataset = load_dataset(dataset_name, split=dataset_split)
    tokenized = raw_dataset.map(
        lambda batch: {"input_ids": [tokenize_example(tokenizer, text) for text in batch["text"]]},
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    flat_tokens: List[int] = []
    for ids in tokenized["input_ids"]:
        flat_tokens.extend(ids)

    total_length = (len(flat_tokens) // block_size) * block_size
    flat_tokens = flat_tokens[:total_length]

    sequences = [flat_tokens[i : i + block_size] for i in range(0, total_length, block_size)]
    if not sequences:
        raise ValueError("No sequences were constructed. Increase dataset size or reduce block size.")

    return ConstantLengthDataset(sequences)


class SimpleDataCollator:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        batch_labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"input_ids": batch_input_ids, "labels": batch_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Hypersphere GPT model on fineweb-edu")
    parser.add_argument("--output_dir", type=str, default="./hypersphere-124m", help="Directory for checkpoints")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/fineweb-edu", help="Dataset identifier")
    parser.add_argument("--dataset_split", type=str, default="train[:0.1%]", help="Dataset split for training")
    parser.add_argument("--block_size", type=int, default=2048, help="Sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every n steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument(
        "--norm_type",
        type=str,
        default="hypersphere",
        choices=[
            "layernorm",
            "rmsnorm",
            "hypersphere",
            "hypersphere_learned_radius",
            "prmsnorm",
        ],
        help="Normalization strategy to use across the network",
    )
    parser.add_argument(
        "--prmsnorm_pct",
        type=float,
        default=0.5,
        help="Fraction of channels used for pRMSNorm (only used when norm_type=prmsnorm)",
    )
    parser.add_argument(
        "--hsnorm_radius",
        type=float,
        default=None,
        help="Optional fixed radius for HyperSphereNorm variants",
    )
    args = parser.parse_args()

    tokenizer = TiktokenTokenizer()

    config = HypersphereGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        dropout=0.0,
        use_qk_norm=True,
        use_qk_norm_scale=True,
        use_rotary_embeddings=True,
        use_peri_ln_attn=True,
        use_peri_ln_mlp=True,
        norm_type=args.norm_type,
        prmsnorm_pct=args.prmsnorm_pct,
        hsnorm_radius=args.hsnorm_radius,
    )

    model = HypersphereGPTForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    dataset = build_dataset(tokenizer, args.block_size, args.dataset, args.dataset_split)
    print(f"Prepared {len(dataset)} sequences of length {args.block_size}")
    data_collator = SimpleDataCollator(args.block_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=5,
        report_to="none",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
