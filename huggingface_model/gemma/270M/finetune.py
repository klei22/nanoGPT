# huggingface_model/gemma/270M/finetune.py
# Prevent GPU OOM on some systems
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# --- Custom Callback for Generating Sample Outputs ---
class SampleOutputCallback(TrainerCallback):
    """A callback that generates a sample translation periodically."""

    def __init__(self, tokenizer, test_sentence="The sun is shining today."):
        self.tokenizer = tokenizer
        self.test_sentence = test_sentence
        self.prompt = (
            f"Translate English to Indonesian:\nEnglish: {self.test_sentence}\nIndonesian: "
        )

    def on_log(self, args, state, control, **kwargs):
        # This callback is triggered every `logging_steps`.
        model = kwargs.get("model")
        if model and state.is_world_process_zero:
            print(f"\n--- Sample output at step {state.global_step} ---")

            # Generate output using the current state of the model
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
            # Use eos_token_id for pad_token_id to avoid warnings
            outputs = model.generate(
                **inputs, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id
            )
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean and print the generated text
            generated_text = decoded_output.replace(self.prompt, "").strip()
            print(f"English: {self.test_sentence}")
            print(f"Generated Indonesian: {generated_text}")
            print("---------------------------------------\n")


class TokenizerMixingCollator:
    """Dynamically alternate tokenizers during training.

    The collator switches between the primary tokenizer (Gemma) and an alternate tokenizer
    according to a linear schedule that ramps the alternate tokenizer usage from
    ``start_ratio`` to ``target_ratio`` over ``ramp_steps`` steps. The TrainerCallback
    below updates the ``current_step`` so batching decisions stay in sync with training
    progress.
    """

    def __init__(
        self,
        primary_tokenizer,
        alternate_tokenizer,
        max_length: int = 128,
        start_ratio: float = 0.0,
        target_ratio: float = 0.5,
        ramp_steps: int = 1000,
    ) -> None:
        self.primary_tokenizer = primary_tokenizer
        self.alternate_tokenizer = alternate_tokenizer
        self.max_length = max_length
        self.start_ratio = start_ratio
        self.target_ratio = target_ratio
        self.ramp_steps = max(ramp_steps, 1)
        self.current_step = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    def _current_alt_ratio(self) -> float:
        progress = min(max(self.current_step, 0), self.ramp_steps) / self.ramp_steps
        return self.start_ratio + (self.target_ratio - self.start_ratio) * progress

    def _select_tokenizer(self):
        alt_prob = self._current_alt_ratio()
        if random.random() < alt_prob:
            return self.alternate_tokenizer
        return self.primary_tokenizer

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        tokenizer = self._select_tokenizer()

        prompts = [
            f"Translate English to Indonesian:\nEnglish: {example['translation']['en']}\nIndonesian: {example['translation']['id']}"
            for example in batch
        ]

        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100

        tokenized["labels"] = labels

        return tokenized


class TokenizerScheduleCallback(TrainerCallback):
    """Keep the collator's notion of training step in sync with Trainer."""

    def __init__(self, collator: TokenizerMixingCollator):
        self.collator = collator

    def on_step_begin(self, args, state, control, **kwargs):
        self.collator.set_step(state.global_step)

# --- Main Script ---
def main(args):
    # 1. Load the dataset
    print("Loading the dataset...")
    # Using a larger portion for more meaningful training
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-id", split="train[:10%]")

    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # 2. Load the tokenizer and model
    model_name = "google/gemma-3-270m"
    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    alternate_tokenizer = AutoTokenizer.from_pretrained(args.alternate_tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager"
    )  # "eager" is the recommended setting for Gemma 270M

    # FIX: Set pad_token to eos_token for decoder-only models
    for tok in (tokenizer, alternate_tokenizer):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    # 3. Build the on-the-fly collator that alternates tokenizers
    data_collator = TokenizerMixingCollator(
        primary_tokenizer=tokenizer,
        alternate_tokenizer=alternate_tokenizer,
        max_length=args.max_length,
        start_ratio=args.alternate_start_ratio,
        target_ratio=args.alternate_target_ratio,
        ramp_steps=args.alternate_ramp_steps or args.total_iterations,
    )

    # 4. Define training arguments from command-line args
    training_args = TrainingArguments(
        output_dir="./gemma-3-270m-opus-100-en-id-causal",
        # Control training with steps instead of epochs
        max_steps=args.total_iterations,
        # Set strategies to 'steps' to use the step-based arguments
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        # Use the provided frequency for logging, eval, and saving
        logging_steps=args.sample_frequency,
        eval_steps=args.sample_frequency,
        save_steps=args.sample_frequency,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    # Instantiate the custom callbacks
    sample_callback = SampleOutputCallback(tokenizer=tokenizer)
    schedule_callback = TokenizerScheduleCallback(collator=data_collator)

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[sample_callback, schedule_callback],  # Add the callbacks here
    )

    # 7. Start training
    print("Starting the training process...")
    trainer.train()

    # Save the final model state
    trainer.save_model()
    print("Training complete! Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Gemma model for translation.")
    parser.add_argument(
        "--total_iterations",
        type=int,
        default=5000,
        help="Total number of training steps (iterations) to perform."
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1000,
        help="How often (in steps) to save, evaluate, and generate a sample output."
    )
    parser.add_argument(
        "--alternate_tokenizer_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Tokenizer to mix in alongside Gemma (e.g., a lighter-weight Mistral tokenizer)."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--alternate_start_ratio",
        type=float,
        default=0.0,
        help="Starting probability of using the alternate tokenizer."
    )
    parser.add_argument(
        "--alternate_target_ratio",
        type=float,
        default=0.5,
        help="Target probability of using the alternate tokenizer once the ramp completes."
    )
    parser.add_argument(
        "--alternate_ramp_steps",
        type=int,
        default=None,
        help="Steps over which to linearly ramp the alternate tokenizer probability (defaults to total_iterations)."
    )
    parsed_args = parser.parse_args()
    main(parsed_args)

