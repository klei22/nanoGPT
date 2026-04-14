"""Build a manual LM-head router for Gemma 270M and evaluate on OPUS-100 en-es.

Base groups:
1) latin: tokens containing Latin-script characters.
2) punct: punctuation-only tokens (Unicode punctuation + common ES/EN punctuation).
3) other: everything else (including byte/special tokens).

Routing modes:
- three_way: latin vs punct vs other.
- latin_punct_vs_other: (latin U punct) vs other.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


COMMON_ES_EN_PUNCT = {
    ".", ",", "!", "?", ":", ";", "'", '"', "-", "—", "–", "(", ")", "[", "]", "{", "}",
    "¡", "¿", "…", "«", "»", "‹", "›", "`", "´", "/", "\\", "|", "@", "#", "&", "*", "%",
    "_",
}
BYTE_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
FEW_SHOT_EXAMPLES = [
    ("Good morning.", "Buenos días."),
    ("Where is the train station?", "¿Dónde está la estación de tren?"),
    ("I would like a glass of water.", "Me gustaría un vaso de agua."),
]


@dataclass
class EvalStats:
    total_target_tokens: int = 0
    top1_correct_routed: int = 0
    top1_correct_full: int = 0
    route_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


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
    return unicodedata.category(ch).startswith("P")


def _is_punctuation_only_text(text: str) -> bool:
    meaningful = [c for c in text if not c.isspace()]
    if not meaningful:
        return False
    return all(_is_common_punctuation(c) for c in meaningful)


def _build_token_groups(tokenizer: AutoTokenizer, vocab_size: int) -> Dict[str, List[int]]:
    groups = {"latin": [], "punct": [], "other": [], "byte": []}
    raw_tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))

    for token_id, raw_token in enumerate(raw_tokens):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if BYTE_TOKEN_RE.match(raw_token or ""):
            groups["byte"].append(token_id)
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
    for key in ("latin", "punct", "other", "byte (subset of other)"):
        src_key = "byte" if key.startswith("byte") else key
        count = len(groups[src_key])
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
    route_groups: Dict[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out = {}
    for key, id_list in route_groups.items():
        if not id_list:
            raise ValueError(f"Routing group '{key}' is empty; cannot compute average prototype.")
        ids = torch.tensor(id_list, dtype=torch.long, device=lm_head_weight.device)
        avg = lm_head_weight.index_select(0, ids).mean(dim=0)
        out[key] = torch.nn.functional.normalize(avg, dim=0).to(device)
    return out


def _router_scores(hidden_last: torch.Tensor, prototypes: Dict[str, torch.Tensor], ordered_names: Sequence[str]) -> torch.Tensor:
    mat = torch.stack([prototypes[name] for name in ordered_names], dim=0)
    return torch.matmul(hidden_last, mat.T)


def _build_route_groups(base_groups: Dict[str, Sequence[int]], route_mode: str) -> Dict[str, List[int]]:
    if route_mode == "three_way":
        return {
            "latin": list(base_groups["latin"]),
            "punct": list(base_groups["punct"]),
            "other": list(base_groups["other"]),
        }
    if route_mode == "latin_punct_vs_other":
        latin_punct = sorted(set(base_groups["latin"]) | set(base_groups["punct"]))
        return {
            "latin_punct": latin_punct,
            "other": list(base_groups["other"]),
        }
    raise ValueError(f"Unknown route_mode: {route_mode}")


def _build_3shot_prompt(english_text: str) -> str:
    lines = [
        "Translate English to Spanish.",
        "",
    ]
    for src, tgt in FEW_SHOT_EXAMPLES:
        lines.append(f"English: {src}")
        lines.append(f"Spanish: {tgt}")
        lines.append("")
    lines.append(f"English: {english_text}")
    lines.append("Spanish:")
    return "\n".join(lines)


def _evaluate_router(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    route_groups: Dict[str, Sequence[int]],
    byte_ids: Sequence[int],
    prototypes: Dict[str, torch.Tensor],
    split: str,
    max_samples: int,
    max_target_tokens: int,
    device: torch.device,
    byte_fallback: bool,
) -> EvalStats:
    stats = EvalStats()
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)
    lm_head_weight = _normalize_rows(model.lm_head.weight.detach())

    route_names = list(route_groups.keys())
    byte_set = set(byte_ids)
    group_ids = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in route_groups.items()
    }

    for ex in ds.select(range(min(max_samples, len(ds)))):
        en = ex["translation"]["en"]
        es = ex["translation"]["es"]
        prompt = _build_3shot_prompt(en)

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
            route = _router_scores(hidden_last, prototypes, route_names).argmax(dim=-1).item()
            chosen = route_names[route]
            stats.route_counts[chosen] += 1

            candidate_ids_list = list(route_groups[chosen])
            if byte_fallback and chosen != "other":
                candidate_ids_list = sorted(set(candidate_ids_list) | byte_set)
            if not candidate_ids_list:
                candidate_ids_list = list(byte_set)
            candidate_ids = torch.tensor(candidate_ids_list, dtype=torch.long, device=device)

            candidate_weight = lm_head_weight.index_select(0, candidate_ids)
            logits = torch.matmul(torch.nn.functional.normalize(hidden_last, dim=-1), candidate_weight.T)
            pred_local = torch.argmax(logits, dim=-1)
            pred_id = candidate_ids[pred_local].item()
            full_logits = torch.matmul(torch.nn.functional.normalize(hidden_last, dim=-1), lm_head_weight.T)
            pred_id_full = torch.argmax(full_logits, dim=-1).item()
            gold_id = target_ids[0, idx].item()

            stats.total_target_tokens += 1
            if pred_id == gold_id:
                stats.top1_correct_routed += 1
            if pred_id_full == gold_id:
                stats.top1_correct_full += 1

            next_gold = target_ids[:, idx : idx + 1]
            running = torch.cat([running, next_gold], dim=1)

    return stats


def _next_token_id(
    hidden_last: torch.Tensor,
    lm_head_weight: torch.Tensor,
    route_names: Sequence[str],
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_set: set[int],
    byte_fallback: bool,
    routed: bool,
    device: torch.device,
) -> int:
    hidden_norm = torch.nn.functional.normalize(hidden_last, dim=-1)
    if not routed:
        logits_full = torch.matmul(hidden_norm, lm_head_weight.T)
        return torch.argmax(logits_full, dim=-1).item()

    route = _router_scores(hidden_last, prototypes, route_names).argmax(dim=-1).item()
    chosen = route_names[route]
    candidate_ids_list = list(route_groups[chosen])
    if byte_fallback and chosen != "other":
        candidate_ids_list = sorted(set(candidate_ids_list) | byte_set)
    if not candidate_ids_list:
        candidate_ids_list = list(byte_set)

    candidate_ids = torch.tensor(candidate_ids_list, dtype=torch.long, device=device)
    candidate_weight = lm_head_weight.index_select(0, candidate_ids)
    logits = torch.matmul(hidden_norm, candidate_weight.T)
    pred_local = torch.argmax(logits, dim=-1)
    return candidate_ids[pred_local].item()


def _generate_translation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    english_text: str,
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_ids: Sequence[int],
    max_new_tokens: int,
    routed: bool,
    byte_fallback: bool,
    device: torch.device,
) -> str:
    prompt = _build_3shot_prompt(english_text)
    running = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)

    route_names = list(route_groups.keys())
    byte_set = set(byte_ids)
    lm_head_weight = _normalize_rows(model.lm_head.weight.detach())
    eos = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(running, output_hidden_states=True, use_cache=False)
        hidden_last = out.hidden_states[-1][:, -1, :]
        next_id = _next_token_id(
            hidden_last=hidden_last,
            lm_head_weight=lm_head_weight,
            route_names=route_names,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_set=byte_set,
            byte_fallback=byte_fallback,
            routed=routed,
            device=device,
        )
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        running = torch.cat([running, next_token], dim=1)
        if eos is not None and next_id == eos:
            break

    full_text = tokenizer.decode(running[0], skip_special_tokens=True)
    if "Spanish:" in full_text:
        return full_text.split("Spanish:", 1)[1].strip()
    return full_text.strip()


def _print_validation_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_ids: Sequence[int],
    example_split: str,
    num_examples: int,
    max_new_tokens: int,
    byte_fallback: bool,
    device: torch.device,
) -> None:
    if num_examples <= 0:
        return
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=example_split)
    count = min(num_examples, len(ds))

    print("\nValidation examples (before=full LM head, after=routed LM head)")
    for idx, ex in enumerate(ds.select(range(count)), start=1):
        en = ex["translation"]["en"]
        es_ref = ex["translation"]["es"]
        pred_full = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=en,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=max_new_tokens,
            routed=False,
            byte_fallback=byte_fallback,
            device=device,
        )
        pred_routed = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=en,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=max_new_tokens,
            routed=True,
            byte_fallback=byte_fallback,
            device=device,
        )
        print(f"\nExample {idx}")
        print(f"EN: {en}")
        print(f"REF: {es_ref}")
        print(f"Before (full): {pred_full}")
        print(f"After  (routed): {pred_routed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Gemma 270M token router eval (OPUS-100 en-es).")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--split", type=str, default="train[:1%]")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_target_tokens", type=int, default=64)
    parser.add_argument(
        "--route_mode",
        type=str,
        default="three_way",
        choices=["three_way", "latin_punct_vs_other"],
        help="Routing mode: 3-way (latin/punct/other) or 2-way ((latin+punct)/other).",
    )
    parser.add_argument(
        "--byte_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include byte tokens in non-other candidate sets as fallback.",
    )
    parser.add_argument("--example_split", type=str, default="validation[:20]")
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--example_max_new_tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    model.to(device)
    model.eval()

    vocab_size = tokenizer.vocab_size
    base_groups = _build_token_groups(tokenizer, vocab_size)
    _print_group_table(base_groups, vocab_size)

    route_groups = _build_route_groups(base_groups, args.route_mode)
    prototypes = _compute_group_prototypes(model.lm_head.weight.detach(), route_groups, device)
    stats = _evaluate_router(
        model=model,
        tokenizer=tokenizer,
        route_groups=route_groups,
        byte_ids=base_groups["byte"],
        prototypes=prototypes,
        split=args.split,
        max_samples=args.max_samples,
        max_target_tokens=args.max_target_tokens,
        device=device,
        byte_fallback=args.byte_fallback,
    )

    acc_routed = 100.0 * stats.top1_correct_routed / max(1, stats.total_target_tokens)
    acc_full = 100.0 * stats.top1_correct_full / max(1, stats.total_target_tokens)
    print("\nRouter evaluation (teacher-forced next-token prediction)")
    print(f"- Routing mode: {args.route_mode}")
    print(f"- Byte fallback: {args.byte_fallback}")
    print(f"- Evaluated tokens: {stats.total_target_tokens}")
    print(f"- Top-1 accuracy without routing (full LM head): {acc_full:.2f}%")
    print(f"- Top-1 accuracy with routing: {acc_routed:.2f}%")
    route_total = max(1, sum(stats.route_counts.values()))
    route_summary = " ".join(
        f"{name}={100.0 * count / route_total:.2f}%" for name, count in stats.route_counts.items()
    )
    print(f"- Route distribution: {route_summary}")

    _print_validation_examples(
        model=model,
        tokenizer=tokenizer,
        route_groups=route_groups,
        prototypes=prototypes,
        byte_ids=base_groups["byte"],
        example_split=args.example_split,
        num_examples=args.num_examples,
        max_new_tokens=args.example_max_new_tokens,
        byte_fallback=args.byte_fallback,
        device=device,
    )


if __name__ == "__main__":
    main()
