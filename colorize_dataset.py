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

import argparse, io, pickle
from pathlib import Path
from typing import Callable, List, Sequence

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


def _colour(ids: List[int], scalars: List[float], decode: Callable[[Sequence[int]], str]) -> Text:
    vals = torch.tensor(scalars, dtype=torch.float32)
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
    out = Text()
    for tid, v in zip(ids, norm):
        r = int((1 - v.item()) * 255); g = int(v.item() * 255)
        out.append(decode([tid]), style=f"bold #{r:02x}{g:02x}00")
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
    p.add_argument(
        "--topk_colwidth",
        type=int,
        default=20,
        help="Max characters for each top-k column; set -1 to disable clipping",
    )
    p.add_argument(
        "--rank_saturation",
        type=int,
        default=100,
        help="Rank value that maps to the reddest colour in rank heatmap",
    )
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
        table.add_column("logit", justify="right", no_wrap=True)
        table.add_column("rank", justify="right", no_wrap=True)
        table.add_column("p_tgt", justify="right", no_wrap=True)
        table.add_column("p_left", justify="right", no_wrap=True)
        col_width = None if args.topk_colwidth < 0 else args.topk_colwidth
        for _ in range(args.topk):
            if col_width is None:
                table.add_column(justify="center", no_wrap=True)
            else:
                table.add_column(justify="center", no_wrap=True, width=col_width)

    while tokens_left > 0:
        # Build window
        seq = data[pos : pos + block + 1]
        if len(seq) < 2:
            break  # not enough tokens to predict next

        ctx_tok = torch.from_numpy(seq[:-1].astype(np.int64))[None].to(args.device)
        with autocast_ctx:
            logits, _ = model(ctx_tok)
        logits = logits.squeeze(0)  # (ctx_len, vocab)
        ctx_len = logits.size(0)
        tgt_token = int(seq[-1])  # ground-truth next token

        if args.display == "token":
            scalar_val = (
                F.softmax(logits[-1], dim=-1)[tgt_token].item()
                if args.mode == "softmax" else logits[-1, tgt_token].item()
            )
            ids.append(tgt_token)
            scalars.append(scalar_val)
        else:
            # probabilities and rankings
            probs = F.softmax(logits[-1], dim=-1)
            tgt_logit = logits[-1, tgt_token].item()
            tgt_prob = probs[tgt_token].item()
            rank = int((logits[-1] > logits[-1, tgt_token]).sum().item()) + 1
            prob_left = probs[logits[-1] > logits[-1, tgt_token]].sum().item()

            # style numeric columns
            rank_norm = (min(rank, args.rank_saturation) - 1) / max(args.rank_saturation - 1, 1)
            rr = int(rank_norm * 255); rg = int((1 - rank_norm) * 255)
            rank_text = Text(str(rank), style=f"bold #{rr:02x}{rg:02x}00")

            rp = int((1 - tgt_prob) * 255); gp = int(tgt_prob * 255)
            prob_text = Text(f"{tgt_prob:.4f}", style=f"bold #{rp:02x}{gp:02x}00")

            rl = int(prob_left * 255); gl = int((1 - prob_left) * 255)
            prob_left_text = Text(f"{prob_left:.4f}", style=f"bold #{rl:02x}{gl:02x}00")

            topv, topi = logits[-1].topk(args.topk)
            norm = (topv - topv.min()) / (topv.max() - topv.min() + 1e-6)
            words: List[Text] = []
            for idx, v in zip(topi.tolist(), norm.tolist()):
                r = int((1 - v) * 255); g = int(v * 255)
                style = f"bold #{r:02x}{g:02x}00"
                if idx == tgt_token:
                    style += " underline"
                words.append(Text(decode([idx]), style=style, overflow="crop"))

            row = [f"{tgt_logit:.4f}", rank_text, prob_text, prob_left_text] + words
            table.add_row(*row)

        # advance
        step = 1 if args.window == "rolling" else ctx_len
        pos += step
        tokens_left -= 1 if args.window == "rolling" else min(ctx_len, tokens_left)

    if args.display == "token":
        coloured = _colour(ids, scalars, decode)
        console.print(coloured)
        lines.append(_ansi(coloured))
    else:
        console.print(table)
        lines.append(_ansi(table))

    if args.output_file:
        Path(args.output_file).write_text("".join(lines), "utf-8", errors="replace")
        console.print(f"[cyan]Saved → {args.output_file}[/cyan]")


if __name__ == "__main__":
    main()
