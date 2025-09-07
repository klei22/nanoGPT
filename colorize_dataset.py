# colorize_dataset.py  (rolling-window mode added)
"""Colourise a dataset split using a trained GPT model.

Modes
-----
* **minmax**  – colour on chosen-token *logits* (after min-max normalisation).
* **softmax** – colour on chosen-token *probabilities*.

Window strategies
-----------------
* **block**   – default.  Chunk the dataset into **non-overlapping** blocks of
  `block_size` tokens.  Fast, but context resets each block.
* **rolling** – slide a *rolling* window: shift by **one token at a time** so
  every prediction gets the longest possible context.  Slower but preserves
  continuity.

Example
-------
```bash
# rolling window, colour 3 000 tokens starting  at offset 50 000
python colorize_dataset.py \
  --out_dir out/my_run \
  --dataset tiny_shakespeare \
  --split train \
  --offset 50000 \
  --num_tokens 3000 \
  --mode softmax \
  --window rolling
```
"""
from __future__ import annotations

import argparse, io, pickle, math
from pathlib import Path
from typing import Callable, List, Sequence, Dict

import numpy as np
import torch, torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.text import Text
import tiktoken  # type: ignore

from model import GPT, GPTConfig

################################################################################
# helpers
################################################################################

def _ansi(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor").print(renderable)
    return buf.getvalue()


def _escape_ws(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

def _colour(ids: List[int], scalars: List[float], decode: Callable[[Sequence[int]], str], escape_ws: bool = True) -> Text:
    vals = torch.tensor(scalars, dtype=torch.float32)
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
    out = Text()
    for tid, v in zip(ids, norm):
        r = int((1 - v.item()) * 255); g = int(v.item() * 255)
        token = decode([tid])
        if escape_ws:
            token = _escape_ws(token)
        out.append(token, style=f"bold #{r:02x}{g:02x}00")
    return out

# byte-fallback helpers -------------------------------------------------------

def _ccwb_encode(text: str, stoi):
    lst: List[int] = []
    for ch in text:
        if ch in stoi:
            lst.append(stoi[ch])
        else:
            lst.extend(stoi[bytes([b])] for b in ch.encode())
    return lst


def _ccwb_decode(ids: List[int], itos):
    out, buf = [], []
    def flush():
        if buf:
            out.append(b"".join(buf).decode("utf-8", "replace")); buf.clear()
    for tok in ids:
        if tok < 256:
            buf.append(itos[tok])
        else:
            flush(); out.append(itos[tok])
    flush(); return "".join(out)

################################################################################
# CLI / loading
################################################################################

def parse_args():
    p = argparse.ArgumentParser("Colourise a dataset split with a trained GPT model")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--ckpt_name", default="ckpt.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--num_tokens", type=int, default=1024)
    p.add_argument("--block_size", type=int)
    p.add_argument("--mode", choices=["minmax", "softmax"], default="minmax")
    p.add_argument("--window", choices=["block", "rolling"], default="block", help="Context window strategy")
    p.add_argument("--offset", type=int, default=0, help="Starting token index within the binary dataset file")
    p.add_argument("--output_file", default="dataset_color.txt")
    p.add_argument("--display", choices=["token", "topk"], default="token", help="Output format")
    p.add_argument("--topk", type=int, default=10, help="Number of top predictions to display when using topk display")
    p.add_argument("--max_token_chars", type=int, default=20, help="Maximum characters for top-k token columns (-1 to disable clipping)")
    p.add_argument("--rank_red", type=int, default=100, help="Rank value treated as fully red in heatmap")
    p.add_argument("--target_style", default="cyan",
                   help="Color to highlight the target token or 'underline' to underline it")
    p.add_argument("--bold_target", action=argparse.BooleanOptionalAction, default=True,
                   help="Bold the target token when highlighting")
    p.add_argument("--escape_whitespace", action=argparse.BooleanOptionalAction, default=True,
                   help="Show newline and tab characters as escape sequences")
    p.add_argument("--plot_metrics", action="store_true", help="Generate Plotly graphs for prediction metrics")
    p.add_argument("--plot_file", default="metrics.html", help="Output HTML file for Plotly metrics")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma for metrics plotting")
    return p.parse_args()


def load_tok(meta: Path):
    meta_obj = pickle.load(meta.open("rb"))
    tk = meta_obj.get("tokenizer"); stoi, itos = meta_obj.get("stoi"), meta_obj.get("itos")
    if tk == "tiktoken":
        enc = tiktoken.get_encoding(meta_obj["tiktoken_encoding"])
        return lambda s: enc.encode(s, allowed_special={""}), lambda l: enc.decode(l)
    if tk == "sentencepiece":
        return lambda s: [stoi[c] for c in s], lambda l: "".join(itos[i] for i in l)
    if tk == "custom_char_with_byte_fallback":
        return lambda s: _ccwb_encode(s, stoi), lambda l: _ccwb_decode(l, itos)
    return lambda s: [stoi[c] for c in s], lambda l: "".join(itos[i] for i in l)

################################################################################
# main
################################################################################

def main():
    args = parse_args(); console = Console()

    # --- load model -----------------------------------------------------------------
    ckpt = torch.load(Path(args.out_dir) / args.ckpt_name, map_location=args.device)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.to(args.device).eval(); torch.set_grad_enabled(False)
    if args.block_size:
        model.update_block_size(args.block_size)

    # --- activation hooks ----------------------------------------------------------
    act_names = ["t0"]
    for i in range(1, model.config.n_layer + 1):
        act_names.extend([f"a{i}", f"m{i}"])
    activations: Dict[str, torch.Tensor] = {}

    def save_activation(name: str):
        def hook(_module, _inp, out):
            activations[name] = out.detach()
        return hook

    model.transformer.drop.register_forward_hook(save_activation("t0"))
    for i, block in enumerate(model.transformer.h, 1):
        block.attn.register_forward_hook(save_activation(f"a{i}"))
        block.mlp.register_forward_hook(save_activation(f"m{i}"))

    # --- tokenizer ------------------------------------------------------------------
    encode, decode = load_tok(Path("data") / args.dataset / "meta.pkl")

    # --- data mm --------------------------------------------------------------------
    dtype = np.uint32 if model.config.vocab_size == 100277 else np.uint16
    data = np.memmap(Path("data") / args.dataset / f"{args.split}.bin", dtype=dtype, mode="r")
    if args.offset >= len(data) - 1:
        raise ValueError("offset beyond dataset length")

    ptd = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptd)
        if "cuda" in args.device else torch.no_grad()
    )

    block = args.block_size or model.config.block_size
    pos = args.offset
    tokens_left = min(args.num_tokens, len(data) - 1 - pos)

    lines: List[str] = []

    if args.display == "token":
        ids: List[int] = []
        scalars: List[float] = []
    else:
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("target", no_wrap=True)
        table.add_column("xent", justify="right", no_wrap=True)
        table.add_column("rank", justify="right", no_wrap=True)
        table.add_column("p_tgt", justify="right", no_wrap=True)
        table.add_column("p_left", justify="right", no_wrap=True)
        for name in act_names:
            table.add_column(name, justify="right", no_wrap=True)
        for _ in range(args.topk):
            table.add_column(justify="center", no_wrap=True)

    if args.plot_metrics:
        metrics_rank: List[float] = []
        metrics_p_left: List[float] = []
        metrics_p_tgt: List[float] = []
        metrics_p_top1: List[float] = []
        metrics_ce: List[float] = []
        metrics_focal: List[float] = []

    while tokens_left > 0:
        # Build window
        seq = data[pos : pos + block + 1]
        if len(seq) < 2:
            break  # not enough tokens to predict next

        activations.clear()
        ctx_tok = torch.from_numpy(seq[:-1].astype(np.int64))[None].to(args.device)
        with autocast_ctx:
            logits, _ = model(ctx_tok)
        logits = logits.squeeze(0)  # (ctx_len, vocab)
        ctx_len = logits.size(0)
        tgt_token = int(seq[-1])  # ground-truth next token

        probs = F.softmax(logits[-1], dim=-1)
        tgt_prob = probs[tgt_token].item()
        tgt_logit = logits[-1, tgt_token].item()
        rank = int((logits[-1] > logits[-1, tgt_token]).sum().item()) + 1
        prob_left = probs[logits[-1] > logits[-1, tgt_token]].sum().item()
        p_top1 = probs.max().item()
        ce = -math.log(tgt_prob + 1e-12)
        focal = ((1 - tgt_prob) ** args.focal_gamma) * ce

        if args.plot_metrics:
            metrics_rank.append(rank)
            metrics_p_left.append(prob_left)
            metrics_p_tgt.append(tgt_prob)
            metrics_p_top1.append(p_top1)
            metrics_ce.append(ce)
            metrics_focal.append(focal)

        if args.display == "token":
            scalar_val = tgt_prob if args.mode == "softmax" else tgt_logit
            ids.append(tgt_token)
            scalars.append(scalar_val)
        else:
            topv, topi = logits[-1].topk(args.topk)
            norm = (topv - topv.min()) / (topv.max() - topv.min() + 1e-6)
            words: List[Text] = []
            for idx, v in zip(topi.tolist(), norm.tolist()):
                r = int((1 - v) * 255); g = int(v * 255)
                style = f"#{r:02x}{g:02x}00"
                if idx == tgt_token:
                    if args.target_style == "underline":
                        if args.bold_target:
                            style += " bold"
                        style += " underline"
                    else:
                        style = args.target_style
                        if args.bold_target:
                            style = f"bold {style}"
                token = decode([idx])
                if args.max_token_chars >= 0:
                    token = token[: args.max_token_chars]
                if args.escape_whitespace:
                    token = _escape_ws(token)
                words.append(Text(token, style=style))

            rank_norm = 1 - (min(rank, args.rank_red) - 1) / max(args.rank_red - 1, 1)
            r = int((1 - rank_norm) * 255); g = int(rank_norm * 255)
            rank_text = Text(str(rank), style=f"bold #{r:02x}{g:02x}00")

            v = tgt_prob
            r = int((1 - v) * 255); g = int(v * 255)
            p_tgt_text = Text(f"{tgt_prob:.4f}", style=f"bold #{r:02x}{g:02x}00")

            v = 1 - prob_left
            r = int((1 - v) * 255); g = int(v * 255)
            p_left_text = Text(f"{prob_left:.4f}", style=f"bold #{r:02x}{g:02x}00")

            target_word = decode([tgt_token])
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

            target_emb = model.transformer.wte.weight[tgt_token].to(torch.float32)
            dots: List[float] = []
            for name in act_names:
                vec = activations[name][0, -1].to(torch.float32)
                dots.append(torch.dot(vec, target_emb).item())
            vals = torch.tensor(dots)
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
            best_idx = int(vals.argmax().item())
            act_texts: List[Text] = []
            for i, (d, v) in enumerate(zip(dots, norm_vals.tolist())):
                r = int((1 - v) * 255); g = int(v * 255)
                style = f"#{r:02x}{g:02x}00"
                if i == best_idx:
                    style = f"bold {style}"
                act_texts.append(Text(f"{d:.2f}", style=style))

            row = [Text(target_word, style=target_style), f"{ce:.4f}", rank_text, p_tgt_text, p_left_text] + act_texts + words
            table.add_row(*row)

        # advance
        step = 1 if args.window == "rolling" else ctx_len
        pos += step
        tokens_left -= 1 if args.window == "rolling" else min(ctx_len, tokens_left)

    if args.display == "token":
        coloured = _colour(ids, scalars, decode, escape_ws=args.escape_whitespace)
        console.print(coloured)
        lines.append(_ansi(coloured))
    else:
        console.print(table)
        lines.append(_ansi(table))

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
