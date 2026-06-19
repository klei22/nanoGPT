"""Analyse Gemma 270M predictions on a validation dataset."""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from contextlib import nullcontext
from typing import List, Sequence

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


def _colour(
    ids: Sequence[int],
    scalars: Sequence[float],
    decode: callable,
    escape_ws: bool = True,
) -> Text:
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


def _ensure_prefix(full: torch.Tensor, prefix: torch.Tensor) -> int:
    """Return the length of the prefix within the full tensor."""
    if prefix.numel() == 0:
        return 0
    plen = prefix.size(0)
    if plen <= full.size(0) and torch.equal(full[:plen], prefix):
        return plen
    # fall back to searching for the prefix (should rarely trigger)
    for start in range(full.size(0) - plen + 1):
        if torch.equal(full[start : start + plen], prefix):
            return start + plen
    return plen


################################################################################
# CLI
################################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Analyse Gemma 270M predictions on a dataset")
    p.add_argument("--model_name", default="google/gemma-3-270m")
    p.add_argument("--dataset_name", default="Helsinki-NLP/opus-100")
    p.add_argument("--dataset_config", default="en-zh")
    p.add_argument(
        "--split",
        default="train[:2%]",
        help="Dataset split to analyse (Hugging Face split notation)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples to analyse",
    )
    p.add_argument(
        "--prompt_template",
        default="Translate English to Chinese:\nEnglish: {source}\nChinese: ",
        help="Template for constructing the prompt. Use {source} placeholder.",
    )
    p.add_argument(
        "--append_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the tokenizer EOS token after the target text.",
    )
    p.add_argument(
        "--display",
        choices=["token", "topk"],
        default="topk",
        help="Display colourised tokens or a rich table of metrics.",
    )
    p.add_argument(
        "--mode",
        choices=["minmax", "softmax"],
        default="minmax",
        help="Colour tokens by logits (minmax) or probabilities (softmax).",
    )
    p.add_argument("--topk", type=int, default=10, help="Top-k predictions to show per token.")
    p.add_argument(
        "--max_token_chars",
        type=int,
        default=20,
        help="Trim decoded tokens to at most this many characters (-1 to disable).",
    )
    p.add_argument(
        "--rank_red",
        type=int,
        default=100,
        help="Rank value treated as fully red in activation heatmaps.",
    )
    p.add_argument(
        "--target_style",
        default="cyan",
        help="Style for highlighting the ground-truth token in top-k tables.",
    )
    p.add_argument(
        "--bold_target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bold the ground-truth token when highlighting.",
    )
    p.add_argument(
        "--escape_whitespace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show newline and tab characters as escape sequences.",
    )
    p.add_argument(
        "--activation_view",
        choices=["target", "rank", "rank_word", "none"],
        default="target",
        help="Per-layer activation statistics to display alongside predictions.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Computation dtype when running on CUDA devices.",
    )
    p.add_argument(
        "--output_file",
        default="",
        help="Optional path to save the rendered analysis.",
    )
    p.add_argument(
        "--plot_metrics",
        action="store_true",
        help="Generate Plotly metrics plots for all analysed tokens.",
    )
    p.add_argument("--plot_file", default="gemma_metrics.html")
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss metric.",
    )
    return p.parse_args()


################################################################################
# main
################################################################################


def main() -> None:
    args = parse_args()
    console = Console()

    console.print("[bold]Loading dataset...[/bold]")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)

    console.print(f"[bold]Loading model {args.model_name}...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model.to(device)
    if device.type == "cuda" and args.dtype != "float32":
        model.to(dtype=dtype)
    model.eval()
    torch.set_grad_enabled(False)

    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

    all_lines: List[str] = []

    output_head = model.get_output_embeddings()
    emb_weight = output_head.weight.detach().to(torch.float32).cpu()

    if args.plot_metrics:
        metrics_rank: List[float] = []
        metrics_p_left: List[float] = []
        metrics_p_tgt: List[float] = []
        metrics_p_top1: List[float] = []
        metrics_ce: List[float] = []
        metrics_focal: List[float] = []

    analysed = 0
    for idx, example in enumerate(dataset):
        if analysed >= args.max_samples:
            break

        translation = example.get("translation")
        if translation is None:
            console.print("[red]Expected `translation` field with `en` and `zh` keys.[/red]")
            return
        source = translation.get("en", "")
        target = translation.get("zh", "")

        prompt = args.prompt_template.format(source=source)
        target_text = target
        if args.append_eos and tokenizer.eos_token:
            target_text = target_text + tokenizer.eos_token
        full_text = prompt + target_text

        enc = tokenizer(full_text, return_tensors="pt")
        prompt_enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        prompt_len = _ensure_prefix(input_ids, prompt_enc["input_ids"][0])

        inputs = {k: v.to(device) for k, v in enc.items()}
        if device.type == "cuda" and args.dtype != "float32":
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
            else:
                autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            outputs = model(**inputs, output_hidden_states=args.activation_view != "none")

        logits = outputs.logits[0].to(torch.float32)  # (seq_len, vocab)
        hidden_states = outputs.hidden_states if args.activation_view != "none" else None

        seq_ids = input_ids.to(torch.long)
        pred_logits = logits[:-1]
        target_ids = seq_ids[1:]
        positions = torch.arange(target_ids.size(0), device=pred_logits.device)
        mask = positions >= (prompt_len - 1)
        pred_logits = pred_logits[mask]
        target_ids = target_ids[mask]
        if hidden_states is not None:
            layer_vectors: List[List[torch.Tensor]] = []
            for layer in hidden_states:
                layer_vectors.append(layer[0, :-1][mask].detach().to(torch.float32).cpu())
        else:
            layer_vectors = []

        pred_logits = pred_logits.detach().to(torch.float32).cpu()
        target_ids = target_ids.detach().cpu()

        if target_ids.numel() == 0:
            continue

        console.print(f"[bold cyan]Sample {analysed + 1}[/bold cyan] – {source}")
        all_lines.append(f"Sample {analysed + 1}: {source}\n")

        if args.display == "token":
            ids = target_ids.tolist()
            if args.mode == "softmax":
                probs = F.softmax(pred_logits, dim=-1)
                scalars = probs[range(len(ids)), target_ids.tolist()].tolist()
            else:
                scalars = pred_logits[range(len(ids)), target_ids.tolist()].tolist()
            coloured = _colour(ids, scalars, decode, escape_ws=args.escape_whitespace)
            console.print(coloured)
            all_lines.append(_ansi(coloured))
        else:
            table = Table(show_header=True, box=None, pad_edge=False)
            table.add_column("token", no_wrap=True)
            table.add_column("xent", justify="right", no_wrap=True)
            table.add_column("rank", justify="right", no_wrap=True)
            table.add_column("p_tgt", justify="right", no_wrap=True)
            table.add_column("p_left", justify="right", no_wrap=True)

            layer_names: List[str] = []
            if layer_vectors:
                layer_names = [f"h{i}" for i in range(len(layer_vectors))]
                for name in layer_names:
                    table.add_column(name, justify="right", no_wrap=True)

            for i in range(args.topk):
                table.add_column(f"top{i+1}", justify="center", no_wrap=True)

            probs = F.softmax(pred_logits, dim=-1)

            if layer_vectors and args.activation_view == "target":
                target_embs = emb_weight[target_ids]
                layer_target_scores: List[torch.Tensor] = []
                layer_target_norms: List[torch.Tensor] = []
                for vectors in layer_vectors:
                    scores = torch.sum(vectors * target_embs, dim=1)
                    layer_target_scores.append(scores)
                    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
                    layer_target_norms.append(norm)
            else:
                layer_target_scores = []
                layer_target_norms = []

            for row_idx, (logit_row, prob_row, tgt_id) in enumerate(
                zip(pred_logits, probs, target_ids)
            ):
                tgt_token = int(tgt_id)
                tgt_prob = float(prob_row[tgt_token])
                tgt_logit = float(logit_row[tgt_token])
                rank = int((logit_row > logit_row[tgt_token]).sum().item()) + 1
                prob_left = float(prob_row[logit_row > logit_row[tgt_token]].sum().item())
                p_top1 = float(prob_row.max().item())
                ce = -math.log(tgt_prob + 1e-12)
                focal = ((1 - tgt_prob) ** args.focal_gamma) * ce

                if args.plot_metrics:
                    metrics_rank.append(rank)
                    metrics_p_left.append(prob_left)
                    metrics_p_tgt.append(tgt_prob)
                    metrics_p_top1.append(p_top1)
                    metrics_ce.append(ce)
                    metrics_focal.append(focal)

                topv, topi = logit_row.topk(args.topk)
                norm = (topv - topv.min()) / (topv.max() - topv.min() + 1e-6)
                words: List[Text] = []
                for idx_tok, v in zip(topi.tolist(), norm.tolist()):
                    r = int((1 - v) * 255)
                    g = int(v * 255)
                    style = f"#{r:02x}{g:02x}00"
                    if idx_tok == tgt_token:
                        if args.target_style == "underline":
                            if args.bold_target:
                                style += " bold"
                            style += " underline"
                        else:
                            style = args.target_style
                            if args.bold_target:
                                style = f"bold {style}"
                    token = decode([idx_tok])
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

                token_text = decode([tgt_token])
                if args.max_token_chars >= 0:
                    token_text = token_text[: args.max_token_chars]
                if args.escape_whitespace:
                    token_text = _escape_ws(token_text)
                target_style = args.target_style if args.target_style != "underline" else "underline"
                if args.bold_target:
                    target_style = f"bold {target_style}" if target_style else "bold"
                token_cell = Text(token_text, style=target_style)

                layer_cells: List[Text] = []
                if layer_vectors:
                    for layer_idx, vectors in enumerate(layer_vectors):
                        if args.activation_view == "target":
                            val = float(layer_target_scores[layer_idx][row_idx])
                            norm_val = float(layer_target_norms[layer_idx][row_idx])
                            rr = int((1 - norm_val) * 255)
                            gg = int(norm_val * 255)
                            layer_cells.append(Text(f"{val:.2f}", style=f"bold #{rr:02x}{gg:02x}00"))
                        elif args.activation_view in {"rank", "rank_word"}:
                            vec = vectors[row_idx]
                            scores = emb_weight @ vec
                            best_id = int(torch.argmax(scores))
                            layer_rank = int((logit_row > logit_row[best_id]).sum().item()) + 1
                            rank_norm_layer = 1 - (
                                min(layer_rank, args.rank_red) - 1
                            ) / max(args.rank_red - 1, 1)
                            rr = int((1 - rank_norm_layer) * 255)
                            gg = int(rank_norm_layer * 255)
                            if args.activation_view == "rank":
                                layer_cells.append(Text(str(layer_rank), style=f"bold #{rr:02x}{gg:02x}00"))
                            else:
                                tok = decode([best_id])
                                if args.max_token_chars >= 0:
                                    tok = tok[: args.max_token_chars]
                                if args.escape_whitespace:
                                    tok = _escape_ws(tok)
                                layer_cells.append(
                                    Text(f"{tok}:{layer_rank}", style=f"bold #{rr:02x}{gg:02x}00")
                                )
                        else:  # none
                            layer_cells.append(Text(""))
                row = [token_cell, f"{ce:.4f}", rank_text, p_tgt_text, p_left_text] + layer_cells + words
                table.add_row(*row)

            console.print(table)
            all_lines.append(_ansi(table))

        analysed += 1
        console.print()

    if args.output_file:
        Path(args.output_file).write_text("".join(all_lines), "utf-8", errors="replace")
        console.print(f"[cyan]Saved analysis → {args.output_file}[/cyan]")

    if args.plot_metrics and analysed:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        x = list(range(len(metrics_rank)))
        fig = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "target rank",
                "prob left",
                "p(target)",
                "p(top1)",
                "cross entropy",
                "focal loss",
            ],
        )
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
