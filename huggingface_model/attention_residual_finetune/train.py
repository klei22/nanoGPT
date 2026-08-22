"""CLI for fine-tuning only the final attention-residual pseudo-query."""

import argparse

import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from huggingface_model.attention_residual_finetune import FinalAttentionResidualCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="out/hf-final-attention-residual")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    model = FinalAttentionResidualCausalLM(base_model)
    print("Trainable parameters:", model.trainable_parameter_names())

    dataset = load_dataset(args.dataset, args.dataset_config)

    def tokenize(batch):
        encoded = tokenizer(
            batch[args.text_column], truncation=True, padding="max_length", max_length=args.max_length
        )
        encoded["labels"] = [
            [token if mask else -100 for token, mask in zip(ids, masks)]
            for ids, masks in zip(encoded["input_ids"], encoded["attention_mask"])
        ]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    Trainer(model=model, args=training_args, train_dataset=tokenized["train"]).train()
    torch_path = f"{args.output_dir}/final_attention_residual.pt"
    torch.save(model.residual.state_dict(), torch_path)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
