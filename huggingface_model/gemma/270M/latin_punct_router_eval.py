"""Build a 3-way manual LM-head router for Gemma 270M and evaluate on OPUS-100 en-es.

Groups:
1) Tokens containing Latin script characters.
2) Punctuation-only tokens (Unicode punctuation / common ES-EN punctuation symbols).
3) Everything else (including byte/special tokens).
"""
from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


COMMON_ES_EN_PUNCT = {
    ".", ",", "!", "?", ":", ";", "'", '"', "-", "—", "–", "(", ")", "[", "]", "{", "}",
    "¡", "¿", "…", "«", "»", "‹", "›", "`", "´", "/", "\\", "|", "@", "#", "&", "*", "%",
    "_",
}


@dataclass
class EvalStats:
    total_target_tokens: int = 0
    top1_correct: int = 0
    route_latin: int = 0
    route_punct: int = 0
    route_other: int = 0


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def _is_latin_char(ch: str) -> bool:
    if not ch:
        return False
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return "LATIN" in name


def _is_common_punctuation(ch: str) -> bool:
    if ch in COMMON_ES_EN_PUNCT:
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P")


def _is_punctuation_only_text(text: str) -> bool:
    meaningful = [c for c in text if not c.isspace()]
    if not meaningful:
        return False
    return all(_is_common_punctuation(c) for c in meaningful)


def _build_token_groups(tokenizer: AutoTokenizer, vocab_size: int) -> Dict[str, List[int]]:
    groups = {"latin": [], "punct": [], "other": []}
    for token_id in range(vocab_size):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if any(_is_latin_char(c) for c in decoded):
            groups["latin"].append(token_id)
        elif _is_punctuation_only_text(decoded):
            groups["punct"].append(token_id)
        else:
            groups["other"].append(token_id)
    return groups


def _print_group_table(groups: Dict[str, Sequence[int]], vocab_size: int) -> None:
    headers = ["Group", "Raw token count", "% of vocab"]
    rows = []
    for key in ("latin", "punct", "other"):
        count = len(groups[key])
        pct = 100.0 * count / vocab_size
        rows.append((key, count, f"{pct:.2f}%"))

    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print("\nToken routing groups for vocabulary")
    print(line)
    print("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(line)
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line)


def _compute_group_prototypes(
    lm_head_weight: torch.Tensor,
    groups: Dict[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out = {}
    for key in ("latin", "punct", "other"):
        ids = torch.tensor(groups[key], dtype=torch.long, device=lm_head_weight.device)
        avg = lm_head_weight.index_select(0, ids).mean(dim=0)
        out[key] = torch.nn.functional.normalize(avg, dim=0).to(device)
    return out


def _router_scores(hidden_last: torch.Tensor, prototypes: Dict[str, torch.Tensor]) -> torch.Tensor:
    mat = torch.stack([prototypes["latin"], prototypes["punct"], prototypes["other"]], dim=0)
    return torch.matmul(hidden_last, mat.T)


def _evaluate_router(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    split: str,
    max_samples: int,
    max_target_tokens: int,
    device: torch.device,
) -> EvalStats:
    stats = EvalStats()
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)
    lm_head_weight = _normalize_rows(model.lm_head.weight.detach())

    group_names = ["latin", "punct", "other"]
    group_ids = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in groups.items()
    }

    for ex in ds.select(range(min(max_samples, len(ds)))):
        en = ex["translation"]["en"]
        es = ex["translation"]["es"]
        prompt = f"Translate English to Spanish:\nEnglish: {en}\nSpanish:"

        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(es, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if target_ids.numel() == 0:
            continue

        steps = min(max_target_tokens, target_ids.size(1))
        running = prompt_ids

        for idx in range(steps):
            with torch.no_grad():
                out = model(running, output_hidden_states=True, use_cache=False)
            hidden_last = out.hidden_states[-1][:, -1, :]  # post-final layernorm state
            route = _router_scores(hidden_last, prototypes).argmax(dim=-1).item()
            chosen = group_names[route]
            if chosen == "latin":
                stats.route_latin += 1
            elif chosen == "punct":
                stats.route_punct += 1
            else:
                stats.route_other += 1

            candidate_ids = group_ids[chosen]
            candidate_weight = lm_head_weight.index_select(0, candidate_ids)
            logits = torch.matmul(torch.nn.functional.normalize(hidden_last, dim=-1), candidate_weight.T)
            pred_local = torch.argmax(logits, dim=-1)
            pred_id = candidate_ids[pred_local].item()
            gold_id = target_ids[0, idx].item()

            stats.total_target_tokens += 1
            if pred_id == gold_id:
                stats.top1_correct += 1

            next_gold = target_ids[:, idx : idx + 1]
            running = torch.cat([running, next_gold], dim=1)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual 3-way Gemma 270M token router eval (OPUS-100 en-es).")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m")
    parser.add_argument("--split", type=str, default="train[:1%]")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_target_tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    model.to(device)
    model.eval()

    vocab_size = tokenizer.vocab_size
    groups = _build_token_groups(tokenizer, vocab_size)
    _print_group_table(groups, vocab_size)

    prototypes = _compute_group_prototypes(model.lm_head.weight.detach(), groups, device)
    stats = _evaluate_router(
        model=model,
        tokenizer=tokenizer,
        groups=groups,
        prototypes=prototypes,
        split=args.split,
        max_samples=args.max_samples,
        max_target_tokens=args.max_target_tokens,
        device=device,
    )

    acc = 100.0 * stats.top1_correct / max(1, stats.total_target_tokens)
    total_routes = max(1, stats.route_latin + stats.route_punct + stats.route_other)
    print("\nRouter evaluation (teacher-forced next-token prediction)")
    print(f"- Evaluated tokens: {stats.total_target_tokens}")
    print(f"- Top-1 accuracy with routed LM head: {acc:.2f}%")
    print(
        "- Route distribution: latin={:.2f}% punct={:.2f}% other={:.2f}%".format(
            100.0 * stats.route_latin / total_routes,
            100.0 * stats.route_punct / total_routes,
            100.0 * stats.route_other / total_routes,
        )
    )


if __name__ == "__main__":
    main()
