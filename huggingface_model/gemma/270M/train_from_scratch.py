# huggingface_model/gemma/270M/train_from_scratch.py
"""Train a Gemma 270M model from scratch on the FineWeb internet corpus."""
import os

# Prevent GPU OOM on some systems
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
import csv
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union, cast

import torch
from datasets import Dataset, IterableDataset, load_dataset, load_dataset_builder
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


@dataclass
class BenchmarkSpec:
    """Configuration for a multiple-choice evaluation benchmark."""

    name: str
    dataset: Union[Dataset, IterableDataset]
    formatter: Callable[[Dict[str, object]], Dict[str, object]]


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


def _ensure_csv(path: str, headers: Iterable[str]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(headers))
            writer.writeheader()


def _format_hellaswag(example: Dict[str, object]) -> Dict[str, object]:
    ctx_a = str(example.get("ctx_a", "")).strip()
    ctx_b = str(example.get("ctx_b", "")).strip()
    context = ctx_a if ctx_b == "" else f"{ctx_a} {ctx_b}".strip()
    endings = example.get("endings")
    if not isinstance(endings, Sequence):
        endings = []
    return {
        "context": context,
        "choices": [str(choice) for choice in endings],
        "label": int(example.get("label", 0)),
    }


def _format_piqa(example: Dict[str, object]) -> Dict[str, object]:
    goal = str(example.get("goal", ""))
    sol1 = str(example.get("sol1", ""))
    sol2 = str(example.get("sol2", ""))
    return {
        "context": goal,
        "choices": [sol1, sol2],
        "label": int(example.get("label", 0)),
    }


def _score_choices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    choices: Sequence[str],
    device: torch.device,
) -> List[float]:
    """Return log-likelihood scores for each choice conditioned on the context."""

    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    scores: List[float] = []
    for choice in choices:
        choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
        if not choice_ids:
            scores.append(float("-inf"))
            continue
        input_ids = torch.tensor([context_ids + choice_ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([[(-100) for _ in context_ids] + choice_ids], device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        loss = outputs.loss.item()
        scores.append(-loss * len(choice_ids))
    return scores


def _evaluate_multiple_choice(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    benchmark: BenchmarkSpec,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> float:
    """Compute accuracy for a multiple-choice benchmark dataset."""

    model_was_training = model.training
    model.eval()

    if max_samples is not None and max_samples < 0:
        max_samples = None

    total = 0
    correct = 0
    iterable: Iterable[Dict[str, object]]
    if isinstance(benchmark.dataset, IterableDataset):
        iterable = benchmark.dataset.take(max_samples) if max_samples else benchmark.dataset
    else:
        iterable = benchmark.dataset

    for example in iterable:
        formatted = benchmark.formatter(example)
        context = str(formatted["context"])
        choices = [str(choice) for choice in cast(Sequence[str], formatted["choices"])]
        label = int(formatted["label"])
        scores = _score_choices(model, tokenizer, context, choices, device)
        predicted = max(range(len(scores)), key=lambda idx: scores[idx])
        correct += int(predicted == label)
        total += 1
        if max_samples is not None and total >= max_samples:
            break

    if model_was_training:
        model.train()

    return correct / total if total else 0.0


class MetricsLoggerCallback(TrainerCallback):
    """Collect and persist evaluation metrics to a CSV file."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        csv_path: str,
        benchmarks: List[BenchmarkSpec],
        max_benchmark_samples: Optional[int],
    ) -> None:
        self.tokenizer = tokenizer
        self.csv_path = csv_path
        self.benchmarks = benchmarks
        self.max_benchmark_samples = max_benchmark_samples
        self.last_train_loss: Optional[float] = None
        self.fieldnames = [
            "step",
            "train_loss",
            "val_loss",
        ] + [f"{bench.name}_accuracy" for bench in benchmarks]
        _ensure_csv(self.csv_path, self.fieldnames)

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        model = kwargs.get("model")
        if model is not None:
            self.model = model  # type: ignore[attr-defined]

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs and "loss" in logs:
            self.last_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return

        metrics = metrics or {}
        val_loss = float(metrics.get("eval_loss", float("nan")))
        model: Optional[AutoModelForCausalLM] = kwargs.get("model")
        if model is None:
            model = getattr(self, "model", None)  # type: ignore[attr-defined]
        if model is None:
            return

        device = next(model.parameters()).device
        row: Dict[str, float] = {
            "step": state.global_step,
            "train_loss": self.last_train_loss if self.last_train_loss is not None else float("nan"),
            "val_loss": val_loss,
        }

        for benchmark in self.benchmarks:
            accuracy = _evaluate_multiple_choice(
                model,
                self.tokenizer,
                benchmark,
                device=device,
                max_samples=self.max_benchmark_samples,
            )
            row[f"{benchmark.name}_accuracy"] = accuracy

        full_row = {name: row.get(name, float("nan")) for name in self.fieldnames}
        with open(self.csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow(full_row)



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

    evaluation_enabled = lm_eval is not None and args.eval_frequency > 0
    evaluation_strategy = "steps" if evaluation_enabled else "no"
    if not evaluation_enabled and args.eval_frequency > 0:
        raise ValueError("Evaluation frequency > 0 requires a non-zero eval_fraction.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.total_iterations,
        logging_strategy="steps",
        evaluation_strategy=evaluation_strategy,
        save_strategy="steps",
        logging_steps=args.log_frequency,
        eval_steps=args.eval_frequency if evaluation_enabled else None,
        save_steps=args.checkpoint_frequency,
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

    print("Preparing benchmark datasets...")
    benchmarks: List[BenchmarkSpec] = []
    if args.disable_benchmarks:
        print("Benchmark evaluations disabled via CLI flag.")
    else:
        hellaswag_split = f"validation[:{args.hellaswag_samples}]" if args.hellaswag_samples else "validation"
        hellaswag = load_dataset("hellaswag", split=hellaswag_split)
        benchmarks.append(BenchmarkSpec("hellaswag", hellaswag, _format_hellaswag))

        if not args.disable_piqa:
            piqa_split = f"validation[:{args.piqa_samples}]" if args.piqa_samples else "validation"
            piqa = load_dataset("piqa", split=piqa_split)
            benchmarks.append(BenchmarkSpec("piqa", piqa, _format_piqa))

    metrics_csv = os.path.join(args.output_dir, "training_metrics.csv")
    benchmark_sample_cap: Optional[int] = None if args.max_benchmark_samples < 0 else args.max_benchmark_samples

    callbacks: List[TrainerCallback] = [
        SampleOutputCallback(tokenizer=tokenizer, prompt=args.sample_prompt),
        MetricsLoggerCallback(
            tokenizer=tokenizer,
            csv_path=metrics_csv,
            benchmarks=benchmarks,
            max_benchmark_samples=benchmark_sample_cap,
        ),
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

    if evaluation_enabled:
        final_metrics = trainer.evaluate()
        print("Final evaluation metrics:", final_metrics)

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
        "--log_frequency",
        type=int,
        default=200,
        help="Steps between logging outputs.",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=1000,
        help="Steps between evaluation runs.",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=1000,
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
        "--disable_benchmarks",
        action="store_true",
        help="Skip external benchmark evaluations during training.",
    )
    parser.add_argument(
        "--disable_piqa",
        action="store_true",
        help="Disable PIQA benchmark evaluation while keeping other benchmarks enabled.",
    )
    parser.add_argument(
        "--hellaswag_samples",
        type=str,
        default="",
        help=(
            "Optional dataset slicing string for the HellaSwag validation split (e.g. '5%')."
        ),
    )
    parser.add_argument(
        "--piqa_samples",
        type=str,
        default="",
        help="Optional dataset slicing string for the PIQA validation split.",
    )
    parser.add_argument(
        "--max_benchmark_samples",
        type=int,
        default=256,
        help="Maximum number of benchmark samples to score per evaluation (use -1 for all).",
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
