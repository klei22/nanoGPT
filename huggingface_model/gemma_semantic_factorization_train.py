#!/usr/bin/env python3
"""Train Gemma 3 270M with semantic tokenizer factorization vs its full vocabulary.

This is an experimental comparison harness: the conventional arm trains a Gemma
architecture from scratch with the original tokenizer head, while the factorized
arm rewrites token IDs into (base token, orthographic feature bits) targets and
uses a smaller LM head plus auxiliary feature heads.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPACE_FEATURE = 1
CAPITALIZED_FEATURE = 2
ALL_CAPS_FEATURE = 4
SPACE_MARKERS = ("▁", "Ġ", " ")
DEFAULT_MODEL = "google/gemma-3-270m"


def _has_case(character: str) -> bool:
    return character.lower() != character.upper()


def _split_space_marker(token: str, marker: str) -> tuple[str, str]:
    if token.startswith(marker):
        return marker, token[len(marker) :]
    return "", token


@dataclass(frozen=True)
class FactorizedToken:
    original_id: int
    base_id: int
    feature_mask: int


@dataclass(frozen=True)
class FactorizedVocabulary:
    base_tokens: tuple[str, ...]
    id_map: tuple[FactorizedToken, ...]
    space_marker: str

    @property
    def original_size(self) -> int:
        return len(self.id_map)

    @property
    def base_size(self) -> int:
        return len(self.base_tokens)

    @property
    def removed(self) -> int:
        return self.original_size - self.base_size

    @property
    def reduction_percent(self) -> float:
        return 100 * self.removed / self.original_size if self.original_size else 0.0


def choose_space_marker(tokens: Iterable[str]) -> str:
    vocabulary = tuple(tokens)
    return max(SPACE_MARKERS, key=lambda marker: sum(token.startswith(marker) for token in vocabulary))


def canonicalize_token(token: str, vocabulary: set[str], space_marker: str) -> tuple[str, int]:
    """Return the base spelling and feature bits for an exactly recoverable token."""
    prefix, text = _split_space_marker(token, space_marker)
    features = 0
    base = token
    if prefix and text and text in vocabulary:
        features |= SPACE_FEATURE
        base = text
        prefix = ""
    if text and text[0].isupper():
        candidate = prefix + text[0].lower() + text[1:]
        if candidate in vocabulary:
            features |= CAPITALIZED_FEATURE
            base = candidate
            text = candidate[len(prefix) :]
    cased_characters = [character for character in text if _has_case(character)]
    if cased_characters and all(character.isupper() for character in cased_characters):
        candidate = prefix + text.lower()
        if candidate in vocabulary:
            features |= ALL_CAPS_FEATURE
            base = candidate
    return base, features


def build_factorized_vocabulary(vocabulary_by_token: dict[str, int], head_size: int | None = None) -> FactorizedVocabulary:
    head_size = head_size if head_size is not None else max(vocabulary_by_token.values()) + 1
    tokens_by_id = {token_id: token for token, token_id in vocabulary_by_token.items() if token_id < head_size}
    for token_id in range(head_size):
        tokens_by_id.setdefault(token_id, f"<reserved:{token_id}>")
    space_marker = choose_space_marker(tokens_by_id.values())
    vocabulary = set(tokens_by_id.values())
    base_to_id: dict[str, int] = {}
    id_map: list[FactorizedToken] = []
    for original_id in range(head_size):
        token = tokens_by_id[original_id]
        base_token, features = canonicalize_token(token, vocabulary, space_marker)
        base_id = base_to_id.setdefault(base_token, len(base_to_id))
        id_map.append(FactorizedToken(original_id, base_id, features))
    base_tokens = tuple(sorted(base_to_id, key=base_to_id.get))
    return FactorizedVocabulary(base_tokens, tuple(id_map), space_marker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemma config/tokenizer to train from scratch")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name passed to datasets.load_dataset")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--split", default="train[:10%]")
    parser.add_argument("--validation-split", default="validation[:10%]")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="eager", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--factorized-only", action="store_true", help="Skip conventional baseline")
    parser.add_argument("--write-token-map", help="Optional JSON path for the factorized id/feature map")
    parser.add_argument("--output-dir", default="gemma_semantic_factorization_benchmark")
    return parser.parse_args()


def convert_original_ids(input_ids, id_map, device):
    import torch

    base_lookup = torch.tensor([entry.base_id for entry in id_map], dtype=torch.long, device=device)
    feature_lookup = torch.tensor([entry.feature_mask for entry in id_map], dtype=torch.long, device=device)
    return base_lookup[input_ids], feature_lookup[input_ids]


def add_feature_embeddings(token_embeddings, feature_embeddings):
    """Add auxiliary embeddings without promoting the model's activation dtype."""
    return token_embeddings + feature_embeddings.to(dtype=token_embeddings.dtype)


def apply_feature_head(head, hidden_states):
    """Run an auxiliary head using the dtype of its parameters."""
    return head(hidden_states.to(dtype=head.weight.dtype))


def update_token_loss_totals(totals: dict[int, list[float]], token_ids, token_losses) -> None:
    """Accumulate sum/count pairs for observed validation target token IDs."""
    for token_id, token_loss in zip(token_ids, token_losses):
        loss = float(token_loss)
        if math.isfinite(loss):
            entry = totals.setdefault(int(token_id), [0.0, 0])
            entry[0] += loss
            entry[1] += 1


def token_loss_records(
    totals: dict[int, list[float]],
    tokens_by_id: dict[int, str],
    factorized_vocabulary: FactorizedVocabulary | None = None,
) -> list[dict]:
    """Convert accumulators into stable, YAML-friendly per-token records."""
    records = [
        {
            "token_id": token_id,
            "token": tokens_by_id.get(token_id, f"<reserved:{token_id}>"),
            "mean_loss": loss_sum / count,
            "count": count,
        }
        for token_id, (loss_sum, count) in sorted(totals.items())
        if count
    ]
    if factorized_vocabulary is not None:
        for record in records:
            mapping = factorized_vocabulary.id_map[record["token_id"]]
            record["base_id"] = mapping.base_id
            record["base_token"] = factorized_vocabulary.base_tokens[mapping.base_id]
            record["feature_mask"] = mapping.feature_mask
    return records


def write_factorized_token_map(path, factorized_vocabulary: FactorizedVocabulary) -> None:
    import json

    document = {
        "base_tokens": list(factorized_vocabulary.base_tokens),
        "space_marker": factorized_vocabulary.space_marker,
        "id_map": [
            {"original_id": entry.original_id, "base_id": entry.base_id, "feature_mask": entry.feature_mask}
            for entry in factorized_vocabulary.id_map
        ],
    }
    with open(path, "w", encoding="utf-8") as token_map_file:
        json.dump(document, token_map_file, ensure_ascii=False, indent=2)


class FactorizedGemmaForCausalLM:
    """Small adapter that adds feature embeddings and auxiliary feature heads."""

    def __init__(self, config, factorized_vocabulary: FactorizedVocabulary, attn_implementation="eager"):
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.nn = nn
        fact_config = config.__class__.from_dict(config.to_dict())
        fact_config.vocab_size = factorized_vocabulary.base_size
        fact_config._attn_implementation = attn_implementation
        self.model = AutoModelForCausalLM.from_config(fact_config)
        hidden_size = fact_config.hidden_size
        self.feature_embedding = nn.Embedding(8, hidden_size)
        self.space_head = nn.Linear(hidden_size, 2)
        self.capitalized_head = nn.Linear(hidden_size, 2)
        self.all_caps_head = nn.Linear(hidden_size, 2)
        self.factorized_vocabulary = factorized_vocabulary

    def parameters(self):
        yield from self.model.parameters()
        yield from self.feature_embedding.parameters()
        yield from self.space_head.parameters()
        yield from self.capitalized_head.parameters()
        yield from self.all_caps_head.parameters()

    def to(self, device):
        self.model.to(device)
        self.feature_embedding.to(device)
        self.space_head.to(device)
        self.capitalized_head.to(device)
        self.all_caps_head.to(device)
        return self

    def train(self):
        for module in (self.model, self.feature_embedding, self.space_head, self.capitalized_head, self.all_caps_head):
            module.train()

    def eval(self):
        for module in (self.model, self.feature_embedding, self.space_head, self.capitalized_head, self.all_caps_head):
            module.eval()

    def token_losses(self, original_ids, attention_mask):
        """Return the mean objective and one combined loss per valid original target."""
        base_ids, feature_masks = convert_original_ids(original_ids, self.factorized_vocabulary.id_map, original_ids.device)
        inputs = base_ids[:, :-1]
        features = feature_masks[:, :-1]
        labels = base_ids[:, 1:].clone()
        label_features = feature_masks[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = -100
        label_features[attention_mask[:, 1:] == 0] = -100
        token_embeddings = self.model.get_input_embeddings()(inputs)
        embeds = add_feature_embeddings(token_embeddings, self.feature_embedding(features))
        outputs = self.model(inputs_embeds=embeds, attention_mask=attention_mask[:, :-1], output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        loss_fct = self.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        base_losses = loss_fct(outputs.logits.reshape(-1, outputs.logits.size(-1)), labels.reshape(-1))
        space_targets = ((label_features & SPACE_FEATURE) > 0).long()
        cap_targets = ((label_features & CAPITALIZED_FEATURE) > 0).long()
        caps_targets = ((label_features & ALL_CAPS_FEATURE) > 0).long()
        space_targets[label_features == -100] = -100
        cap_targets[label_features == -100] = -100
        caps_targets[label_features == -100] = -100
        space_losses = loss_fct(apply_feature_head(self.space_head, hidden).reshape(-1, 2), space_targets.reshape(-1))
        cap_losses = loss_fct(apply_feature_head(self.capitalized_head, hidden).reshape(-1, 2), cap_targets.reshape(-1))
        caps_losses = loss_fct(apply_feature_head(self.all_caps_head, hidden).reshape(-1, 2), caps_targets.reshape(-1))
        combined_losses = base_losses + space_losses + cap_losses + caps_losses
        valid_targets = original_ids[:, 1:][attention_mask[:, 1:].bool()]
        valid_losses = combined_losses[labels.reshape(-1) != -100]
        return valid_losses.mean(), valid_targets, valid_losses

    def train_step(self, original_ids, attention_mask):
        loss, _, _ = self.token_losses(original_ids, attention_mask)
        return loss


def count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def conventional_token_losses(model, original_ids, attention_mask):
    import torch.nn.functional as functional

    outputs = model(input_ids=original_ids, attention_mask=attention_mask)
    labels = original_ids[:, 1:]
    valid = attention_mask[:, 1:].bool()
    losses = functional.cross_entropy(
        outputs.logits[:, :-1].reshape(-1, outputs.logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    return losses[valid].mean(), labels[valid], losses[valid]


def validation_batches(input_ids, attention_masks, batch_size, limit):
    for batch_index, start in enumerate(range(0, len(input_ids), batch_size)):
        if batch_index >= limit:
            break
        yield input_ids[start : start + batch_size], attention_masks[start : start + batch_size]


def evaluate_arm(name, model, input_ids, attention_masks, args, tokens_by_id, factorized_vocabulary):
    import torch

    model.eval()
    totals: dict[int, list[float]] = {}
    loss_sum = 0.0
    target_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch, mask in validation_batches(
            input_ids, attention_masks, args.batch_size, args.validation_batches
        ):
            batch = batch.to(args.resolved_device)
            mask = mask.to(args.resolved_device)
            if name == "conventional":
                _, target_ids, token_losses = conventional_token_losses(model, batch, mask)
            else:
                _, target_ids, token_losses = model.token_losses(batch, mask)
            cpu_ids = target_ids.detach().cpu().tolist()
            cpu_losses = token_losses.detach().float().cpu().tolist()
            update_token_loss_totals(totals, cpu_ids, cpu_losses)
            loss_sum += sum(cpu_losses)
            target_count += len(cpu_losses)
    elapsed = time.perf_counter() - started
    model.train()
    mean_loss = loss_sum / target_count if target_count else float("nan")
    return {
        "validation_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "tokens": target_count,
        "seconds": elapsed,
        "tokens_per_second": target_count / elapsed if elapsed else 0.0,
        "token_losses": token_loss_records(
            totals, tokens_by_id, factorized_vocabulary if name == "factorized" else None
        ),
    }


class BenchmarkLogger:
    """Persist an evolving YAML benchmark report and validation plots."""

    def __init__(self, output_dir, metadata):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / "benchmark.yaml"
        self.document = {"metadata": metadata, "validation": []}
        self._write_yaml()

    def record(self, arm, step, training, validation):
        entry = {"arm": arm, "step": step, "training": training, **validation}
        self.document["validation"].append(entry)
        self._write_yaml()
        self._write_plots(entry)

    def _write_yaml(self):
        import yaml

        temporary_path = self.report_path.with_suffix(".yaml.tmp")
        with open(temporary_path, "w", encoding="utf-8") as report_file:
            yaml.safe_dump(self.document, report_file, allow_unicode=True, sort_keys=False)
        os.replace(temporary_path, self.report_path)

    def _write_plots(self, latest_entry):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(8, 5))
        for arm in sorted({entry["arm"] for entry in self.document["validation"]}):
            entries = [entry for entry in self.document["validation"] if entry["arm"] == arm]
            axis.plot([entry["step"] for entry in entries], [entry["validation_loss"] for entry in entries], marker="o", label=arm)
        axis.set(xlabel="training step", ylabel="validation loss", title="Gemma semantic factorization benchmark")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "validation_loss.png", dpi=160)
        plt.close(fig)

        frequent = sorted(latest_entry["token_losses"], key=lambda record: record["count"], reverse=True)[:30]
        if frequent:
            frequent.reverse()
            fig, axis = plt.subplots(figsize=(10, 8))
            labels = [f"{record['token_id']}: {record['token']!r}" for record in frequent]
            axis.barh(labels, [record["mean_loss"] for record in frequent])
            axis.set(xlabel="mean validation loss", title=f"{latest_entry['arm']} token losses at step {latest_entry['step']}")
            fig.tight_layout()
            fig.savefig(
                self.output_dir / f"{latest_entry['arm']}_token_loss_step_{latest_entry['step']:06d}.png",
                dpi=160,
            )
            plt.close(fig)


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.validation_interval <= 0 or args.validation_batches <= 0:
        raise ValueError("validation interval and validation batches must be positive")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    args.resolved_device = device
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model)
    vocabulary = {token: token_id for token, token_id in tokenizer.get_vocab().items() if token_id < config.vocab_size}
    factorized_vocabulary = build_factorized_vocabulary(vocabulary, config.vocab_size)
    print(
        f"factorized vocab: before={factorized_vocabulary.original_size:,} "
        f"after={factorized_vocabulary.base_size:,} removed={factorized_vocabulary.removed:,} "
        f"reduction={factorized_vocabulary.reduction_percent:.2f}%"
    )

    if args.write_token_map:
        write_factorized_token_map(args.write_token_map, factorized_vocabulary)

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    validation_dataset = load_dataset(args.dataset, args.dataset_config, split=args.validation_split)
    texts = [row[args.text_field] for row in dataset if row[args.text_field].strip()]
    validation_texts = [
        row[args.text_field] for row in validation_dataset if row[args.text_field].strip()
    ]
    encoded = tokenizer(
        texts, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt"
    )
    validation_encoded = tokenizer(
        validation_texts,
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    batches = encoded["input_ids"]
    masks = encoded["attention_mask"]
    validation_ids = validation_encoded["input_ids"]
    validation_masks = validation_encoded["attention_mask"]

    arms = []
    if not args.factorized_only:
        conventional_config = config.__class__.from_dict(config.to_dict())
        conventional_config._attn_implementation = args.attn_implementation
        conventional = AutoModelForCausalLM.from_config(conventional_config).to(device)
        arms.append(("conventional", conventional, torch.optim.AdamW(conventional.parameters(), lr=args.lr)))
    factorized = FactorizedGemmaForCausalLM(
        config, factorized_vocabulary, args.attn_implementation
    ).to(device)
    arms.append(("factorized", factorized, torch.optim.AdamW(factorized.parameters(), lr=args.lr)))
    for name, model, _ in arms:
        print(f"{name} parameters={count_parameters(model):,}")

    tokens_by_id = {token_id: token for token, token_id in vocabulary.items()}
    logger = BenchmarkLogger(
        args.output_dir,
        {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "train_split": args.split,
            "validation_split": args.validation_split,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "validation_interval": args.validation_interval,
            "validation_batches": args.validation_batches,
            "seed": args.seed,
            "device": device,
            "attention_implementation": args.attn_implementation,
            "vocabulary_before": factorized_vocabulary.original_size,
            "vocabulary_after": factorized_vocabulary.base_size,
        },
    )

    for name, model, optimizer in arms:
        parameter_count = count_parameters(model)
        validation = evaluate_arm(
            name, model, validation_ids, validation_masks, args, tokens_by_id, factorized_vocabulary
        )
        logger.record(name, 0, {"mean_loss": None, "tokens_per_second": None, "parameters": parameter_count}, validation)
        print(f"{name} validation step=0 loss={validation['validation_loss']:.4f}")
        interval_loss = 0.0
        interval_tokens = 0
        interval_steps = 0
        interval_started = time.perf_counter()
        for step in range(args.steps):
            start = (step * args.batch_size) % max(1, len(batches) - args.batch_size + 1)
            batch = batches[start : start + args.batch_size].to(device)
            attention_mask = masks[start : start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            if name == "conventional":
                labels = batch.clone()
                labels[attention_mask == 0] = -100
                loss = model(input_ids=batch, attention_mask=attention_mask, labels=labels).loss
            else:
                loss = model.train_step(batch, attention_mask)
            loss.backward()
            optimizer.step()
            interval_loss += loss.item()
            interval_tokens += int(attention_mask[:, 1:].sum().item())
            interval_steps += 1
            print(f"{name} step={step + 1} loss={loss.item():.4f}")
            completed_step = step + 1
            if completed_step % args.validation_interval == 0 or completed_step == args.steps:
                training_seconds = time.perf_counter() - interval_started
                validation = evaluate_arm(
                    name, model, validation_ids, validation_masks, args, tokens_by_id, factorized_vocabulary
                )
                training = {
                    "mean_loss": interval_loss / interval_steps,
                    "tokens": interval_tokens,
                    "seconds": training_seconds,
                    "tokens_per_second": interval_tokens / training_seconds if training_seconds else 0.0,
                    "parameters": parameter_count,
                }
                logger.record(name, completed_step, training, validation)
                print(
                    f"{name} validation step={completed_step} "
                    f"loss={validation['validation_loss']:.4f} perplexity={validation['perplexity']:.2f}"
                )
                interval_loss = 0.0
                interval_tokens = 0
                interval_steps = 0
                interval_started = time.perf_counter()


if __name__ == "__main__":
    main()
