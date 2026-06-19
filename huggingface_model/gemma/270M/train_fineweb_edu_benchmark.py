# huggingface_model/gemma/270M/train_fineweb_edu_benchmark.py
"""Train Gemma 270M from scratch on FineWeb-Edu and run a benchmark sweep."""
import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# Prevent GPU OOM on some systems
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "offline")


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


class EvalLossHistoryCallback(TrainerCallback):
    """Collect evaluation loss metrics for plotting."""

    def __init__(self) -> None:
        self.steps: List[int] = []
        self.losses: List[float] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if metrics and "eval_loss" in metrics:
            self.steps.append(state.global_step)
            self.losses.append(metrics["eval_loss"])


class EmbeddingRMSNormWrapper(torch.nn.Module):
    """Apply RMSNorm directly after the token embedding lookup."""

    def __init__(self, embedding: torch.nn.Module, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.embedding = embedding
        self.norm = torch.nn.RMSNorm(hidden_size, eps=eps)

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.embedding(input_ids)
        return self.norm(inputs_embeds)


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


def _get_hidden_size(config: AutoConfig) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Unable to determine hidden size from the model configuration.")


def _plot_validation_losses(steps: List[int], losses: List[float], output_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, marker="o", color="#1f77b4")
    ax.set_title("Validation Loss Over Training")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_validation_comparison(
    baseline: Tuple[List[int], List[float]],
    rmsnorm: Tuple[List[int], List[float]],
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(baseline[0], baseline[1], marker="o", label="Baseline", linewidth=2)
    ax.plot(rmsnorm[0], rmsnorm[1], marker="s", label="Input RMSNorm", linewidth=2)
    ax.set_title("Validation Loss vs. Training Steps")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _run_benchmark_sweep(script_path: str, output_dir: str) -> None:
    print(f"Running benchmark sweep: {script_path}")
    subprocess.run([sys.executable, script_path], check=True, cwd=output_dir)


def _load_benchmark_row(csv_path: str, variant: str) -> Optional[Dict[str, float]]:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Softmax Variant") == variant:
                return {k: float(v) for k, v in row.items() if k != "Softmax Variant"}
    return None


def _plot_benchmark_comparison(
    baseline_csv: str,
    rmsnorm_csv: str,
    output_path: str,
    variant: str = "softmax",
) -> bool:
    baseline = _load_benchmark_row(baseline_csv, variant)
    rmsnorm = _load_benchmark_row(rmsnorm_csv, variant)
    if not baseline or not rmsnorm:
        print("Benchmark CSVs missing or variant not found; skipping benchmark comparison plot.")
        return False

    def parse_block(label: str) -> int:
        return int(label.replace("Block Size ", ""))

    block_labels = sorted(baseline.keys(), key=parse_block)
    baseline_values = [baseline[label] for label in block_labels]
    rmsnorm_values = [rmsnorm[label] for label in block_labels]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(block_labels, baseline_values, marker="o", label="Baseline", linewidth=2)
    ax.plot(block_labels, rmsnorm_values, marker="s", label="Input RMSNorm", linewidth=2)
    ax.set_title(f"Benchmark Sweep (Forward Pass) - {variant}")
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Avg Forward Pass Time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


@dataclass
class RunArtifacts:
    output_dir: str
    eval_steps: List[int]
    eval_losses: List[float]


def _run_training(args: argparse.Namespace, apply_input_rmsnorm: bool, run_label: str) -> RunArtifacts:
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

    if apply_input_rmsnorm:
        hidden_size = _get_hidden_size(config)
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        wrapped_embeddings = EmbeddingRMSNormWrapper(model.get_input_embeddings(), hidden_size, rms_eps)
        model.set_input_embeddings(wrapped_embeddings)

    output_dir = os.path.join(args.output_dir, run_label)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading FineWeb-Edu dataset...")
    dataset_config: Optional[str]
    if args.dataset_config:
        dataset_config = args.dataset_config
    else:
        dataset_config = None

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", dataset_config, split=args.dataset_split)

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
        output_dir=output_dir,
        max_steps=args.total_iterations,
        logging_strategy="steps",
        eval_strategy="steps" if lm_eval is not None else "no",
        save_strategy="steps",
        logging_steps=args.eval_interval,
        eval_steps=args.eval_interval if lm_eval is not None else None,
        save_steps=args.save_interval,
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

    eval_loss_callback = EvalLossHistoryCallback()
    callbacks = [
        SampleOutputCallback(tokenizer=tokenizer, prompt=args.sample_prompt),
        eval_loss_callback,
    ]

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
    print("Training complete! Model saved to", output_dir)

    if eval_loss_callback.steps:
        plot_path = os.path.join(output_dir, args.plot_filename)
        _plot_validation_losses(eval_loss_callback.steps, eval_loss_callback.losses, plot_path)
        print("Saved validation loss plot to", plot_path)
    else:
        print("No evaluation losses recorded; skipping plot generation.")

    if not args.skip_benchmark_sweep:
        _run_benchmark_sweep(args.benchmark_script, output_dir)

    return RunArtifacts(
        output_dir=output_dir,
        eval_steps=eval_loss_callback.steps,
        eval_losses=eval_loss_callback.losses,
    )


def main(args: argparse.Namespace) -> None:
    if not args.single_run:
        os.makedirs(args.output_dir, exist_ok=True)
        baseline = _run_training(args, apply_input_rmsnorm=False, run_label="baseline")
        rmsnorm = _run_training(args, apply_input_rmsnorm=True, run_label="input_rmsnorm")

        if baseline.eval_steps and rmsnorm.eval_steps:
            comparison_path = os.path.join(args.output_dir, args.comparison_plot_filename)
            _plot_validation_comparison(
                (baseline.eval_steps, baseline.eval_losses),
                (rmsnorm.eval_steps, rmsnorm.eval_losses),
                comparison_path,
            )
            print("Saved validation loss comparison plot to", comparison_path)
        else:
            print("Skipping validation loss comparison plot due to missing eval history.")

        benchmark_plot = os.path.join(args.output_dir, args.benchmark_plot_filename)
        if _plot_benchmark_comparison(
            os.path.join(baseline.output_dir, "forward_pass_timing_results.csv"),
            os.path.join(rmsnorm.output_dir, "forward_pass_timing_results.csv"),
            benchmark_plot,
        ):
            print("Saved benchmark comparison plot to", benchmark_plot)
    else:
        _run_training(args, apply_input_rmsnorm=args.apply_input_rmsnorm, run_label="single")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Gemma 270M model from scratch on FineWeb-Edu and run a benchmark sweep."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Split of FineWeb-Edu to load (HF dataset slicing syntax).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="",
        help="Optional configuration name for FineWeb-Edu (empty for default).",
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
        default=100_000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Steps between validation loss measurements.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10_000,
        help="Steps between checkpoint saves.",
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
        "--sample_prompt",
        type=str,
        default="The internet is a vast",
        help="Prompt used for periodic generation samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemma-3-270m-fineweb-edu-from-scratch",
        help="Directory to store checkpoints and final model.",
    )
    parser.add_argument(
        "--plot_filename",
        type=str,
        default="validation_loss.png",
        help="Filename for the validation loss plot.",
    )
    parser.add_argument(
        "--comparison_plot_filename",
        type=str,
        default="validation_loss_comparison.png",
        help="Filename for the validation loss comparison plot.",
    )
    parser.add_argument(
        "--apply_input_rmsnorm",
        action="store_true",
        help="Apply RMSNorm directly after the token embedding lookup.",
    )
    parser.add_argument(
        "--single_run",
        action="store_true",
        help="Run only one training job (controlled by --apply_input_rmsnorm).",
    )
    parser.add_argument(
        "--benchmark_plot_filename",
        type=str,
        default="benchmark_comparison.png",
        help="Filename for the benchmark comparison plot.",
    )
    parser.add_argument(
        "--benchmark_script",
        type=str,
        default="benchmarks/softmax_sweep.py",
        help="Path to the benchmark sweep script to run after training.",
    )
    parser.add_argument(
        "--skip_benchmark_sweep",
        action="store_true",
        help="Skip running the benchmark sweep after training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
