"""Fine-tune SmolLM2's final attention residual on GSM8K training data."""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .model import SmolLM2FinalAttentionResidual

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="out/smollm2-135m-final-attention-residual")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    adapter = SmolLM2FinalAttentionResidual(model)
    print("Trainable parameters:", adapter.trainable_parameter_names)

    train_data = load_dataset("openai/gsm8k", "main", split="train")

    def tokenize(example):
        user_prompt = f"Question: {example['question']}\nAnswer:"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = f"{prompt} {example['answer']}{tokenizer.eos_token}"
        encoded = tokenizer(full_text, truncation=True, padding="max_length", max_length=args.max_length)
        prompt_length = len(tokenizer(prompt, truncation=True, max_length=args.max_length)["input_ids"])
        encoded["labels"] = [
            token if index >= prompt_length and mask else -100
            for index, (token, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"]))
        ]
        return encoded

    tokenized = train_data.map(tokenize, remove_columns=train_data.column_names)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    Trainer(model=model, args=training_args, train_dataset=tokenized).train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    adapter.save(f"{args.output_dir}/final_attention_residual.pt")
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
