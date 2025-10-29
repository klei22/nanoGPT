# huggingface_model/gemma/270M/train_from_scratch.py
"""Train the Gemma 270M model from scratch using Hugging Face Trainer.

This script mirrors the fine-tuning workflow but downloads the pretrained
Gemma checkpoint only to reuse its configuration/tokenizer before
reinitialising all model weights with a Gaussian distribution. A selection of
custom loss functions from ``train_variations.loss_variants`` is available via
command line arguments, along with a suite of typical training hyper-parameters.
Optional text benchmarks can be executed every evaluation interval to monitor
qualitative progress.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from train_variations.loss_variants import (
    attenuated_correct_top1_loss,
    cross_entropy_loss,
    distance_attenuated_top1_loss,
    focal_loss,
    skip_correct_top1_loss,
)

try:
    from benchmarks.dataset_metrics import run_all as run_dataset_metrics
except Exception:
    run_dataset_metrics = None


# Ensure CUDA memory fragmentation mitigation matches the fine-tune script.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Loss registry utilities
# ---------------------------------------------------------------------------
LossFn = Callable[..., torch.Tensor]


@dataclass
class LossConfig:
    name: str
    fn: LossFn
    arg_names: List[str]


LOSS_REGISTRY: Dict[str, LossConfig] = {
    "cross_entropy": LossConfig("cross_entropy", cross_entropy_loss, []),
    "skip_top1": LossConfig("skip_top1", skip_correct_top1_loss, []),
    "distance_attenuated_top1": LossConfig("distance_attenuated_top1", distance_attenuated_top1_loss, ["strength"]),
    "attenuated_correct_top1": LossConfig("attenuated_correct_top1", attenuated_correct_top1_loss, ["attenuation"]),
    "focal": LossConfig("focal", focal_loss, ["gamma"]),
}


# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------
class SampleOutputCallback(TrainerCallback):
    """Generate sample text periodically for qualitative inspection."""

    def __init__(self, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 64):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def _generate(self, model: torch.nn.Module) -> str:
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if text.startswith(self.prompt):
            text = text[len(self.prompt) :]
        return text.strip()

    def on_log(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if not model or not state.is_world_process_zero:
            return
        print(f"\n--- Sample output at step {state.global_step} ---")
        print(self._generate(model))
        print("---------------------------------------\n")


class BenchmarkCallback(TrainerCallback):
    """Run optional benchmarks every evaluation interval."""

    def __init__(self, tokenizer: AutoTokenizer, prompts: List[str], max_new_tokens: int):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if run_dataset_metrics is None:
            print("Benchmark metrics unavailable: `pyspellchecker` not installed.")
            return

        model = kwargs.get("model")
        if model is None:
            return

        print("\n=== Benchmark Metrics ===")
        for prompt in self.prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            metrics = run_dataset_metrics(text)
            print(f"Prompt: {prompt}")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            print("---------------------------")
        print("===========================\n")


# ---------------------------------------------------------------------------
# Trainer subclass to plug in the custom loss functions
# ---------------------------------------------------------------------------
class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_config: LossConfig, loss_kwargs: Dict[str, float], **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.loss_kwargs = loss_kwargs

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Expected `labels` in inputs for language modelling.")

        outputs = model(**inputs)
        logits = outputs.logits

        # Convert Hugging Face ignore index (-100) into the value expected by
        # the loss variants (-1) while keeping a copy for logging.
        targets = labels.clone()
        targets = targets.masked_fill(targets == -100, -1)

        iter_num = int(self.state.global_step)
        loss = self.loss_config.fn(logits, targets, iter_num=iter_num, **self.loss_kwargs)

        if return_outputs:
            return loss, outputs
        return loss


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def reinitialize_model_weights(model: torch.nn.Module, std: float, mean: float) -> None:
    """Apply normal initialisation to the model weights."""

    def _init(module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=mean, std=std)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    model.apply(_init)


def create_loss_kwargs(loss_name: str, args: argparse.Namespace) -> Dict[str, float]:
    if loss_name == "focal":
        return {"gamma": args.focal_gamma}
    if loss_name == "distance_attenuated_top1":
        return {"strength": args.distance_strength}
    if loss_name == "attenuated_correct_top1":
        return {"attenuation": args.correct_top1_attenuation}
    return {}


def bool_flag(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Loading tokenizer and model for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    print("Reinitialising model weights...")
    reinitialize_model_weights(model, std=args.init_std, mean=args.init_mean)

    if args.compile_model:
        if hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        else:
            print("torch.compile not available; continuing without compilation.")

    def preprocess(examples):
        texts = [
            f"Translate English to Chinese:\nEnglish: {ex['en']}\nChinese: {ex['zh']}"
            for ex in examples["translation"]
        ]
        return tokenizer(texts, truncation=True, max_length=args.max_length, padding="max_length")

    print("Tokenizing dataset...")
    remove_columns = train_dataset.column_names
    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=remove_columns,
    )
    tokenized_eval = eval_dataset.map(
        preprocess,
        batched=True,
        remove_columns=remove_columns,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_strategy = "steps" if args.eval_interval > 0 else "no"
    logging_steps = args.log_interval if args.log_interval > 0 else args.eval_interval
    if logging_steps <= 0:
        logging_steps = 50

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        save_strategy="steps" if args.save_interval > 0 else "no",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=args.eval_interval if args.eval_interval > 0 else None,
        save_steps=args.save_interval if args.save_interval > 0 else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        torch_compile=args.compile_model,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_workers,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to=args.report_to if args.report_to else None,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
    )

    # Older versions of ``transformers`` used ``evaluation_strategy`` while newer
    # releases renamed the argument to ``eval_strategy``. Introspect the
    # signature at runtime so the script remains compatible across versions.
    try:
        init_params = TrainingArguments.__init__.__code__.co_varnames
    except AttributeError:
        init_params = ()
    key = "evaluation_strategy" if "evaluation_strategy" in init_params else "eval_strategy"
    training_args_kwargs[key] = eval_strategy

    training_args = TrainingArguments(**training_args_kwargs)

    loss_config = LOSS_REGISTRY[args.loss_function]
    loss_kwargs = create_loss_kwargs(args.loss_function, args)
    print(f"Using loss function: {loss_config.name} with params {loss_kwargs}")

    callbacks: List[TrainerCallback] = []
    if args.sample_prompt:
        callbacks.append(
            SampleOutputCallback(
                tokenizer=tokenizer,
                prompt=args.sample_prompt,
                max_new_tokens=args.sample_max_new_tokens,
            )
        )
    if args.run_benchmarks and args.benchmark_prompts:
        callbacks.append(
            BenchmarkCallback(
                tokenizer=tokenizer,
                prompts=args.benchmark_prompts,
                max_new_tokens=args.benchmark_max_new_tokens,
            )
        )

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval if args.eval_interval > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        loss_config=loss_config,
        loss_kwargs=loss_kwargs,
    )

    print("Starting training from scratch...")
    trainer.train()
    trainer.save_model()
    print("Training complete. Model saved to:", args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Gemma 270M from scratch with custom losses.")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m", help="Hugging Face model identifier.")
    parser.add_argument("--dataset_name", type=str, default="Helsinki-NLP/opus-100", help="Dataset repository to use.")
    parser.add_argument("--dataset_config", type=str, default="en-zh", help="Dataset configuration (language pair).")
    parser.add_argument("--dataset_split", type=str, default="train[:10%]", help="Split of the dataset to use.")
    parser.add_argument("--eval_fraction", type=float, default=0.1, help="Fraction of data reserved for evaluation.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length during tokenization.")
    parser.add_argument("--output_dir", type=str, default="./gemma-3-270m-scratch", help="Directory for checkpoints.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Per-device batch size for evaluation.")
    parser.add_argument("--max_steps", type=int, default=5000, help="Number of training steps to run.")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval in training steps (0 disables eval).")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval in steps (0 disables saving).")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval in steps (<=0 defaults to eval interval).")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Number of warm-up steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of LR scheduler (Hugging Face options).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--dataloader_workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--gradient_checkpointing", type=bool_flag, default=False, help="Enable gradient checkpointing.")
    parser.add_argument("--compile_model", type=bool_flag, default=False, help="Compile the model with torch.compile.")
    parser.add_argument("--fp16", type=bool_flag, default=torch.cuda.is_available(), help="Use fp16 mixed precision.")
    parser.add_argument("--bf16", type=bool_flag, default=False, help="Use bfloat16 mixed precision.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--report_to", type=str, nargs="*", default=None, help="Integration(s) to report metrics to (e.g. wandb).")

    parser.add_argument("--loss_function", type=str, choices=list(LOSS_REGISTRY.keys()), default="cross_entropy", help="Loss function to use.")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for focal loss.")
    parser.add_argument("--distance_strength", type=float, default=0.0, help="Strength for distance attenuated top1 loss.")
    parser.add_argument("--correct_top1_attenuation", type=float, default=1.0, help="Attenuation for attenuated correct top1 loss.")

    parser.add_argument("--init_std", type=float, default=0.02, help="Standard deviation for weight initialisation.")
    parser.add_argument("--init_mean", type=float, default=0.0, help="Mean for weight initialisation.")

    parser.add_argument("--sample_prompt", type=str, default="Translate English to Chinese:\nEnglish: The sun is shining today.\nChinese:", help="Prompt used for periodic sample generation.")
    parser.add_argument("--sample_max_new_tokens", type=int, default=64, help="Number of tokens to generate for samples.")

    parser.add_argument("--run_benchmarks", type=bool_flag, default=False, help="Run benchmark metrics at evaluation intervals.")
    parser.add_argument(
        "--benchmark_prompts",
        type=str,
        nargs="*",
        default=["Translate English to Chinese:\nEnglish: The cat sat on the mat.\nChinese:"],
        help="Prompts to evaluate during benchmark runs.",
    )
    parser.add_argument("--benchmark_max_new_tokens", type=int, default=64, help="Tokens to generate for benchmark prompts.")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
