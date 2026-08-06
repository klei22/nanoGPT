#!/usr/bin/env python3
"""Train Gemma 3 270M with semantic tokenizer factorization vs its full vocabulary.

This is an experimental comparison harness: the conventional arm trains a Gemma
architecture from scratch with the original tokenizer head, while the factorized
arm rewrites token IDs into (base token, orthographic feature bits) targets and
uses a smaller LM head plus auxiliary feature heads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    parser.add_argument("--split", default="train[:1%]")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--factorized-only", action="store_true", help="Skip conventional baseline")
    parser.add_argument("--write-token-map", help="Optional JSON path for the factorized id/feature map")
    return parser.parse_args()


def convert_original_ids(input_ids, id_map, device):
    import torch

    base_lookup = torch.tensor([entry.base_id for entry in id_map], dtype=torch.long, device=device)
    feature_lookup = torch.tensor([entry.feature_mask for entry in id_map], dtype=torch.long, device=device)
    return base_lookup[input_ids], feature_lookup[input_ids]


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

    def __init__(self, config, factorized_vocabulary: FactorizedVocabulary):
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.nn = nn
        fact_config = config.__class__.from_dict(config.to_dict())
        fact_config.vocab_size = factorized_vocabulary.base_size
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

    def train_step(self, original_ids, attention_mask):
        base_ids, feature_masks = convert_original_ids(original_ids, self.factorized_vocabulary.id_map, original_ids.device)
        inputs = base_ids[:, :-1]
        features = feature_masks[:, :-1]
        labels = base_ids[:, 1:].clone()
        label_features = feature_masks[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = -100
        label_features[attention_mask[:, 1:] == 0] = -100
        embeds = self.model.get_input_embeddings()(inputs) + self.feature_embedding(features)
        outputs = self.model(inputs_embeds=embeds, attention_mask=attention_mask[:, :-1], output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        loss_fct = self.nn.CrossEntropyLoss(ignore_index=-100)
        base_loss = loss_fct(outputs.logits.reshape(-1, outputs.logits.size(-1)), labels.reshape(-1))
        space_targets = ((label_features & SPACE_FEATURE) > 0).long()
        cap_targets = ((label_features & CAPITALIZED_FEATURE) > 0).long()
        caps_targets = ((label_features & ALL_CAPS_FEATURE) > 0).long()
        space_targets[label_features == -100] = -100
        cap_targets[label_features == -100] = -100
        caps_targets[label_features == -100] = -100
        space_loss = loss_fct(self.space_head(hidden).reshape(-1, 2), space_targets.reshape(-1))
        cap_loss = loss_fct(self.capitalized_head(hidden).reshape(-1, 2), cap_targets.reshape(-1))
        caps_loss = loss_fct(self.all_caps_head(hidden).reshape(-1, 2), caps_targets.reshape(-1))
        return base_loss + space_loss + cap_loss + caps_loss


def count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
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
    texts = [row[args.text_field] for row in dataset if row[args.text_field].strip()]
    encoded = tokenizer(texts, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt")
    batches = encoded["input_ids"]
    masks = encoded["attention_mask"]

    arms = []
    if not args.factorized_only:
        conventional = AutoModelForCausalLM.from_config(config).to(device)
        arms.append(("conventional", conventional, torch.optim.AdamW(conventional.parameters(), lr=args.lr)))
    factorized = FactorizedGemmaForCausalLM(config, factorized_vocabulary).to(device)
    arms.append(("factorized", factorized, torch.optim.AdamW(factorized.parameters(), lr=args.lr)))
    for name, model, _ in arms:
        print(f"{name} parameters={count_parameters(model):,}")

    for name, model, optimizer in arms:
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
            print(f"{name} step={step + 1} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
