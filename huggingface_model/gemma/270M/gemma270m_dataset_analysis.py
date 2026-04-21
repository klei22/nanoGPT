"""Analyze Gemma 270M predictions on a validation dataset.

Features borrowed from:
* colorize_dataset.py (token heatmaps, top-k tables, metric plots)
* finetune.py (Gemma 270M loading, dataset prompt format)
"""
# Prevent GPU OOM on some systems
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

################################################################################
# helpers
################################################################################

def _ansi(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor").print(renderable)
    return buf.getvalue()


def _escape_ws(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _colour(ids: List[int], scalars: List[float], decode, escape_ws: bool = True) -> Text:
    vals = torch.tensor(scalars, dtype=torch.float32)
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
    out = Text()
    for tid, v in zip(ids, norm):
        r = int((1 - v.item()) * 255)
        g = int(v.item() * 255)
        token = decode([tid])
        if escape_ws:
            token = _escape_ws(token)
        out.append(token, style=f"bold #{r:02x}{g:02x}00")
    return out


def _select_dataset(args):
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if args.split in dataset:
        split = dataset[args.split]
    elif "train" in dataset:
        split_parts = dataset["train"].train_test_split(test_size=args.validation_split, seed=args.seed)
        split = split_parts["test"] if args.split in {"validation", "val", "test"} else split_parts["train"]
    else:
        raise ValueError(f"Split '{args.split}' not found and dataset has no train split.")
    if args.max_examples:
        split = split.select(range(min(args.max_examples, len(split))))
    return split


def _build_prompt(source: str) -> str:
    return f"Translate English to Chinese:\nEnglish: {source}\nChinese: "


def _extract_pair(example, source_lang: str, target_lang: str) -> Tuple[str, str]:
    translation = example.get("translation")
    if not translation or source_lang not in translation or target_lang not in translation:
        raise KeyError("Expected a 'translation' field with source/target languages.")
    return translation[source_lang], translation[target_lang]


def _iter_metrics(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    mask_start: int,
) -> Iterable[Tuple[int, int, float, float, int, float]]:
    probs = F.softmax(logits, dim=-1)
    labels = input_ids[1:]
    for idx in range(mask_start, labels.size(0)):
        tgt_token = labels[idx].item()
        tgt_prob = probs[idx, tgt_token].item()
        tgt_logit = logits[idx, tgt_token].item()
        rank = int((logits[idx] > logits[idx, tgt_token]).sum().item()) + 1
        prob_left = probs[idx, logits[idx] > logits[idx, tgt_token]].sum().item()
        yield idx, tgt_token, tgt_prob, tgt_logit, rank, prob_left

################################################################################
# main
################################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Analyze Gemma 270M predictions on a validation dataset.")
    p.add_argument("--dataset_name", default="Helsinki-NLP/opus-100")
    p.add_argument("--dataset_config", default="en-zh")
    p.add_argument("--split", default="validation")
    p.add_argument("--validation_split", type=float, default=0.1)
    p.add_argument("--source_lang", default="en")
    p.add_argument("--target_lang", default="zh")
    p.add_argument("--model_name", default="google/gemma-3-270m")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max_examples", type=int, default=3)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--mode", choices=["minmax", "softmax"], default="softmax")
    p.add_argument("--display", choices=["token", "topk"], default="token")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--max_token_chars", type=int, default=20)
    p.add_argument("--rank_red", type=int, default=100)
    p.add_argument("--target_style", default="cyan")
    p.add_argument("--bold_target", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--escape_whitespace", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--plot_metrics", action="store_true")
    p.add_argument("--plot_file", default="gemma270m_metrics.html")
    p.add_argument("--output_file", default="gemma270m_dataset_analysis.txt")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--sample_generation", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    console = Console()

    dataset = _select_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(args.device).eval()

    ptd = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptd)
        if "cuda" in args.device else torch.no_grad()
    )

    lines: List[str] = []

    if args.plot_metrics:
        metrics_rank: List[float] = []
        metrics_p_left: List[float] = []
        metrics_p_tgt: List[float] = []
        metrics_p_top1: List[float] = []
        metrics_ce: List[float] = []
        metrics_focal: List[float] = []

    for idx, example in enumerate(dataset):
        source, target = _extract_pair(example, args.source_lang, args.target_lang)
        prompt = _build_prompt(source)
        full_text = f"{prompt}{target}"

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        if len(full_ids) < 2:
            continue
        if len(full_ids) > args.max_length:
            full_ids = full_ids[: args.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        mask_start = max(prompt_len - 1, 0)

        input_ids = torch.tensor(full_ids, device=args.device, dtype=torch.long)

        with torch.no_grad(), autocast_ctx:
            logits = model(input_ids.unsqueeze(0)).logits.squeeze(0)

        logits = logits[:-1]
        if args.display == "token":
            ids: List[int] = []
            scalars: List[float] = []
            for tok_idx, tgt_token, tgt_prob, tgt_logit, rank, prob_left in _iter_metrics(input_ids, logits, mask_start):
                scalar_val = tgt_prob if args.mode == "softmax" else tgt_logit
                ids.append(tgt_token)
                scalars.append(scalar_val)
                if args.plot_metrics:
                    ce = -math.log(tgt_prob + 1e-12)
                    metrics_rank.append(rank)
                    metrics_p_left.append(prob_left)
                    metrics_p_tgt.append(tgt_prob)
                    metrics_p_top1.append(torch.softmax(logits[tok_idx], dim=-1).max().item())
                    metrics_ce.append(ce)
                    metrics_focal.append(((1 - tgt_prob) ** args.focal_gamma) * ce)

            header = Text(f"Example {idx + 1}", style="bold magenta")
            console.print(header)
            coloured = _colour(ids, scalars, tokenizer.decode, escape_ws=args.escape_whitespace)
            console.print(coloured)
            lines.append(_ansi(header))
            lines.append(_ansi(coloured))
        else:
            table = Table(show_header=True, box=None, pad_edge=False)
            table.add_column("target", no_wrap=True)
            table.add_column("xent", justify="right", no_wrap=True)
            table.add_column("rank", justify="right", no_wrap=True)
            table.add_column("p_tgt", justify="right", no_wrap=True)
            table.add_column("p_left", justify="right", no_wrap=True)
            for i in range(args.topk):
                table.add_column(f"top{i+1}", justify="center", no_wrap=True)

            for tok_idx, tgt_token, tgt_prob, tgt_logit, rank, prob_left in _iter_metrics(input_ids, logits, mask_start):
                topv, topi = logits[tok_idx].topk(args.topk)
                norm = (topv - topv.min()) / (topv.max() - topv.min() + 1e-6)
                words: List[Text] = []
                for tok_id, v in zip(topi.tolist(), norm.tolist()):
                    r = int((1 - v) * 255)
                    g = int(v * 255)
                    style = f"#{r:02x}{g:02x}00"
                    if tok_id == tgt_token:
                        if args.target_style == "underline":
                            if args.bold_target:
                                style += " bold"
                            style += " underline"
                        else:
                            style = args.target_style
                            if args.bold_target:
                                style = f"bold {style}"
                    token = tokenizer.decode([tok_id])
                    if args.max_token_chars >= 0:
                        token = token[: args.max_token_chars]
                    if args.escape_whitespace:
                        token = _escape_ws(token)
                    words.append(Text(token, style=style))

                rank_norm = 1 - (min(rank, args.rank_red) - 1) / max(args.rank_red - 1, 1)
                r = int((1 - rank_norm) * 255)
                g = int(rank_norm * 255)
                rank_text = Text(str(rank), style=f"bold #{r:02x}{g:02x}00")

                v = tgt_prob
                r = int((1 - v) * 255)
                g = int(v * 255)
                p_tgt_text = Text(f"{tgt_prob:.4f}", style=f"bold #{r:02x}{g:02x}00")

                v = 1 - prob_left
                r = int((1 - v) * 255)
                g = int(v * 255)
                p_left_text = Text(f"{prob_left:.4f}", style=f"bold #{r:02x}{g:02x}00")

                target_word = tokenizer.decode([tgt_token])
                if args.max_token_chars >= 0:
                    target_word = target_word[: args.max_token_chars]
                if args.escape_whitespace:
                    target_word = _escape_ws(target_word)
                if args.target_style == "underline":
                    target_style = "underline"
                else:
                    target_style = args.target_style
                if args.bold_target:
                    target_style = f"bold {target_style}" if target_style else "bold"

                ce = -math.log(tgt_prob + 1e-12)
                row = [Text(target_word, style=target_style), f"{ce:.4f}", rank_text, p_tgt_text, p_left_text] + words
                table.add_row(*row)

                if args.plot_metrics:
                    metrics_rank.append(rank)
                    metrics_p_left.append(prob_left)
                    metrics_p_tgt.append(tgt_prob)
                    metrics_p_top1.append(torch.softmax(logits[tok_idx], dim=-1).max().item())
                    metrics_ce.append(ce)
                    metrics_focal.append(((1 - tgt_prob) ** args.focal_gamma) * ce)

            header = Text(f"Example {idx + 1}", style="bold magenta")
            console.print(header)
            console.print(table)
            lines.append(_ansi(header))
            lines.append(_ansi(table))

        if args.sample_generation:
            console.print("[bold cyan]Sample generation[/bold cyan]")
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = decoded.replace(prompt, "").strip()
            console.print(f"English: {source}")
            console.print(f"Generated Chinese: {generated_text}")

    if args.output_file:
        Path(args.output_file).write_text("".join(lines), "utf-8", errors="replace")
        console.print(f"[cyan]Saved → {args.output_file}[/cyan]")

    if args.plot_metrics:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        x = list(range(len(metrics_rank)))
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                            subplot_titles=["target rank", "prob left", "p(target)",
                                            "p(top1)", "cross entropy", "focal loss"])
        fig.add_trace(go.Scatter(x=x, y=metrics_rank, name="rank"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_p_left, name="p_left"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_p_tgt, name="p_tgt"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_p_top1, name="p_top1"), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_ce, name="cross_entropy"), row=5, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_focal, name="focal"), row=6, col=1)
        fig.update_layout(height=300 * 6, showlegend=False)
        fig.write_html(args.plot_file)
        console.print(f"[cyan]Saved plot → {args.plot_file}[/cyan]")


if __name__ == "__main__":
    main()
