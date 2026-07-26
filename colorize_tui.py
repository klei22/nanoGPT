"""
Interactive viewer for `colorize_dataset` predictions.

This tool keeps the original data extraction path (dataset + checkpoint) and
presents the results in a small Textual TUI with separate panes for:

* token-by-token summary
* top-k predictions for the selected token
* activation summaries (with toggles for attn / mlp / residual components and
  the existing activation sub-modes)

Usage example
-------------
```
python colorize_tui.py \
  --out_dir out/my_run \
  --dataset tiny_shakespeare \
  --split val \
  --num_tokens 64 \
  --topk 8
```

Note: `textual` is required for the UI (`pip install textual`).
"""
from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Checkbox,
    DataTable,
    Footer,
    Header,
    Label,
    RadioButton,
    RadioSet,
)
import tiktoken  # type: ignore

from model import GPT, GPTConfig

################################################################################
# helpers
################################################################################


@dataclass
class ActivationSummary:
    label: str
    target_score: float
    best_token: str
    best_token_id: int
    rank: int


@dataclass
class TopKEntry:
    token: str
    prob: float
    logit: float
    is_target: bool


@dataclass
class StepResult:
    index: int
    pos: int
    target_token: str
    target_id: int
    cross_entropy: float
    rank: int
    p_left: float
    p_tgt: float
    topk: List[TopKEntry]
    activations: List[ActivationSummary]


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
            out.append(b"".join(buf).decode("utf-8", "replace"))
            buf.clear()

    for tok in ids:
        if tok < 256:
            buf.append(itos[tok])
        else:
            flush()
            out.append(itos[tok])
    flush()
    return "".join(out)


def _escape_ws(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def load_tok(meta: Path):
    meta_obj = pickle.load(meta.open("rb"))
    tk = meta_obj.get("tokenizer")
    stoi, itos = meta_obj.get("stoi"), meta_obj.get("itos")
    if tk == "tiktoken":
        enc = tiktoken.get_encoding(meta_obj["tiktoken_encoding"])
        return lambda s: enc.encode(s, allowed_special={""}), lambda l: enc.decode(l)
    if tk == "sentencepiece":
        return lambda s: [stoi[c] for c in s], lambda l: "".join(itos[i] for i in l)
    if tk == "custom_char_with_byte_fallback":
        return lambda s: _ccwb_encode(s, stoi), lambda l: _ccwb_decode(l, itos)
    return lambda s: [stoi[c] for c in s], lambda l: "".join(itos[i] for i in l)


def _colour_style(value: float, low_is_red: bool = True) -> Style:
    v = max(0.0, min(1.0, value))
    if not low_is_red:
        v = 1 - v
    r = int((1 - v) * 255)
    g = int(v * 255)
    return Style(color=f"#{r:02x}{g:02x}00", bold=True)


################################################################################
# data extraction
################################################################################


def parse_args():
    p = argparse.ArgumentParser("Interactive TUI for colorize_dataset top-k view")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--ckpt_name", default="ckpt.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--num_tokens", type=int, default=64)
    p.add_argument("--block_size", type=int)
    p.add_argument("--window", choices=["block", "rolling"], default="block", help="Context window strategy")
    p.add_argument("--offset", type=int, default=0, help="Starting token index within the binary dataset file")
    p.add_argument("--topk", type=int, default=8, help="Number of top predictions to store for the UI")
    p.add_argument("--max_token_chars", type=int, default=16, help="Maximum characters to show for tokens (-1 to disable)")
    p.add_argument("--escape_whitespace", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--activation_components",
        choices=["wte", "attn", "mlp", "resid"],
        nargs="+",
        default=["wte", "attn", "mlp", "resid"],
        help="Activation components to capture (UI can toggle visibility)",
    )
    return p.parse_args()


def _iter_vectors(activations, include_resid: bool, n_layers: int):
    if "wte" in activations:
        yield "t0", activations["wte"]
    for i in range(n_layers):
        if len(activations["attn"]) > i:
            yield f"a{i+1}", activations["attn"][i]
        if include_resid and len(activations["ar"]) > i:
            yield f"ar{i+1}", activations["ar"][i]
        if len(activations["mlp"]) > i:
            yield f"m{i+1}", activations["mlp"][i]
        if include_resid and len(activations["mr"]) > i:
            yield f"mr{i+1}", activations["mr"][i]


def _collect_activations(model: GPT, activations, logits_row, tgt_token: int, decode, max_token_chars: int, escape_ws: bool) -> List[ActivationSummary]:
    summaries: List[ActivationSummary] = []
    emb = model.lm_head.weight.detach()
    for label, vec in _iter_vectors(activations, include_resid=True, n_layers=model.config.n_layer):
        vec_f = vec.float()
        target_score = torch.dot(vec_f, emb[tgt_token].float()).item()
        scores = emb @ vec_f
        best_tok = torch.argmax(scores).item()
        rank = int((logits_row > logits_row[best_tok]).sum().item()) + 1
        token = decode([best_tok])
        if max_token_chars >= 0:
            token = token[: max_token_chars]
        if escape_ws:
            token = _escape_ws(token)
        summaries.append(
            ActivationSummary(
                label=label,
                target_score=target_score,
                best_token=token,
                best_token_id=best_tok,
                rank=rank,
            )
        )
    return summaries


def _component_for_label(label: str) -> str:
    if label == "t0":
        return "wte"
    if label.startswith("a") and not label.startswith("ar"):
        return "attn"
    if label.startswith("m") and not label.startswith("mr"):
        return "mlp"
    return "resid"


def _gather_predictions(args) -> tuple[list[StepResult], Callable[[Sequence[int]], str]]:
    console = Console()
    ckpt = torch.load(Path(args.out_dir) / args.ckpt_name, map_location=args.device)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.to(args.device).eval()
    torch.set_grad_enabled(False)
    if args.block_size:
        model.update_block_size(args.block_size)

    encode, decode = load_tok(Path("data") / args.dataset / "meta.pkl")

    dtype = np.uint32 if model.config.vocab_size == 100277 else np.uint16
    data = np.memmap(Path("data") / args.dataset / f"{args.split}.bin", dtype=dtype, mode="r")
    if args.offset >= len(data) - 1:
        raise ValueError("offset beyond dataset length")

    ptd = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptd)
        if "cuda" in args.device
        else torch.no_grad()
    )

    block = args.block_size or model.config.block_size
    pos = args.offset
    tokens_left = min(args.num_tokens, len(data) - 1 - pos)

    results: List[StepResult] = []

    while tokens_left > 0:
        seq = data[pos : pos + block + 1]
        if len(seq) < 2:
            break
        ctx_tok = torch.from_numpy(seq[:-1].astype(np.int64))[None].to(args.device)

        activations = {"wte": None, "attn": [], "mlp": [], "ar": [], "mr": []}
        handles = []

        def t0_hook(module, inp, out):
            activations["wte"] = out[0, -1, :].detach()

        handles.append(model.transformer.drop.register_forward_hook(t0_hook))

        for blk in model.transformer.h:
            def make_attn_hook(blk):
                def hook(module, inp, out):
                    outp = out
                    if getattr(blk, "use_peri_ln_attn", False):
                        outp = blk.peri_ln_attn(outp)
                    if getattr(blk, "attn_resid_scaler", None):
                        outp = blk.attn_resid_scaler(outp)
                    activations["attn"].append(outp[0, -1, :].detach())
                return hook

            def make_mlp_hook(blk):
                def hook(module, inp, out):
                    outp = out
                    if getattr(blk, "use_peri_ln_mlp", False):
                        outp = blk.peri_ln_mlp(outp)
                    if getattr(blk, "mlp_resid_scaler", None):
                        outp = blk.mlp_resid_scaler(outp)
                    activations["mlp"].append(outp[0, -1, :].detach())
                return hook

            handles.append(blk.attn.register_forward_hook(make_attn_hook(blk)))
            handles.append(blk.mlp.register_forward_hook(make_mlp_hook(blk)))

        with autocast_ctx:
            logits, _ = model(ctx_tok)

        for h in handles:
            h.remove()

        if activations["wte"] is not None and "resid" in args.activation_components:
            resid = activations["wte"].float().clone()
            for a, m in zip(activations["attn"], activations["mlp"]):
                resid = resid + a.float()
                activations["ar"].append(resid.clone())
                resid = resid + m.float()
                activations["mr"].append(resid.clone())

        logits = logits.squeeze(0)
        ctx_len = logits.size(0)
        tgt_token = int(seq[-1])

        probs = F.softmax(logits[-1], dim=-1)
        tgt_prob = probs[tgt_token].item()
        rank = int((logits[-1] > logits[-1, tgt_token]).sum().item()) + 1
        prob_left = probs[logits[-1] > logits[-1, tgt_token]].sum().item()
        ce = -math.log(tgt_prob + 1e-12)

        token_text = decode([tgt_token])
        if args.max_token_chars >= 0:
            token_text = token_text[: args.max_token_chars]
        if args.escape_whitespace:
            token_text = _escape_ws(token_text)

        topv, topi = logits[-1].topk(args.topk)
        topk: List[TopKEntry] = []
        for logit_val, idx in zip(topv.tolist(), topi.tolist()):
            tok = decode([idx])
            if args.max_token_chars >= 0:
                tok = tok[: args.max_token_chars]
            if args.escape_whitespace:
                tok = _escape_ws(tok)
            topk.append(
                TopKEntry(
                    token=tok,
                    prob=float(probs[idx].item()),
                    logit=float(logit_val),
                    is_target=idx == tgt_token,
                )
            )

        activation_summaries = _collect_activations(
            model,
            activations,
            logits[-1],
            tgt_token,
            decode,
            args.max_token_chars,
            args.escape_whitespace,
        )

        results.append(
            StepResult(
                index=len(results),
                pos=pos,
                target_token=token_text,
                target_id=tgt_token,
                cross_entropy=ce,
                rank=rank,
                p_left=prob_left,
                p_tgt=tgt_prob,
                topk=topk,
                activations=activation_summaries,
            )
        )

        step = 1 if args.window == "rolling" else ctx_len
        pos += step
        tokens_left -= 1 if args.window == "rolling" else min(ctx_len, tokens_left)

        if len(results) >= args.num_tokens:
            break

    console.print(f"[cyan]Captured {len(results)} tokens for interactive viewing.[/cyan]")
    return results, decode


################################################################################
# UI
################################################################################


class ColorizeViewer(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        layout: horizontal;
        height: 1fr;
    }
    #controls {
        width: 32%;
        min-width: 30;
        border: heavy $background 40%;
        padding: 1 1;
    }
    #content {
        layout: vertical;
        width: 1fr;
        border: heavy $background 20%;
    }
    #topk_panel, #activations_panel {
        height: 1fr;
        border: tall $primary;
        padding: 0 1;
    }
    DataTable {
        height: 1fr;
    }
    """

    activation_view: reactive[str] = reactive("rank_word")

    def __init__(self, results: List[StepResult]):
        super().__init__()
        self.results = results
        self.selected_components = {"wte", "attn", "mlp", "resid"}
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="controls"):
                yield Label("Pane controls", id="controls_label")
                yield Label("Activation view")
                with RadioSet(id="activation_view"):
                    yield RadioButton("Rank + token", value=True, id="rank_word")
                    yield RadioButton("Rank only", id="rank")
                    yield RadioButton("Target dot", id="target")
                    yield RadioButton("Hide activations", id="none")
                yield Label("Components")
                yield Checkbox("Token embedding (t0)", value=True, id="wte")
                yield Checkbox("Attention outputs", value=True, id="attn")
                yield Checkbox("Residual stream", value=True, id="resid")
                yield Checkbox("MLP outputs", value=True, id="mlp")
            with Vertical(id="content"):
                self.token_table = DataTable(cursor_type="row", id="tokens_table")
                self.topk_table = DataTable(id="topk_table")
                self.activations_table = DataTable(id="activations_table")
                yield self.token_table
                with Vertical(id="topk_panel"):
                    yield Label("Top-k predictions (right-click/enter to move)")
                    yield self.topk_table
                with Vertical(id="activations_panel"):
                    yield Label("Activation summaries (toggle panes on the left)")
                    yield self.activations_table
        yield Footer()

    def on_mount(self) -> None:
        self._setup_tables()
        self._populate_tokens()
        self._refresh_detail()

    def _setup_tables(self):
        self.token_table.clear(columns=True)
        self.token_table.add_columns("idx", "token", "rank", "p(tgt)", "p_left", "CE")
        self.topk_table.clear(columns=True)
        self.topk_table.add_columns("token", "logit", "prob")
        self.activations_table.clear(columns=True)
        self.activations_table.add_columns("component", "value")
        self.token_table.cursor_type = "row"

    def _populate_tokens(self):
        for res in self.results:
            self.token_table.add_row(
                str(res.index),
                res.target_token,
                str(res.rank),
                f"{res.p_tgt:.4f}",
                f"{res.p_left:.4f}",
                f"{res.cross_entropy:.3f}",
            )
        if self.results:
            self.token_table.focus()
            self.token_table.cursor_coordinate = (0, 0)

    def _filtered_components(self, res: StepResult) -> Iterable[ActivationSummary]:
        for summary in res.activations:
            if _component_for_label(summary.label) in self.selected_components:
                yield summary

    def _activation_display(self, summary: ActivationSummary) -> Text:
        view = self.activation_view
        if view == "target":
            normed = summary.target_score
            return Text(f"{summary.target_score:.2f}", style=_colour_style(normed))
        rank_norm = 1 - (min(summary.rank, 100) - 1) / 99
        style = _colour_style(rank_norm, low_is_red=False)
        if view == "rank":
            return Text(str(summary.rank), style=style)
        elif view == "rank_word":
            return Text(f"{summary.best_token}:{summary.rank}", style=style)
        return Text("–")

    def _refresh_detail(self):
        if not self.results:
            return
        res = self.results[self.selected_index]

        self.topk_table.clear(rows=True)
        for entry in res.topk:
            style = Style(bold=True, color="cyan") if entry.is_target else None
            self.topk_table.add_row(entry.token, f"{entry.logit:.2f}", f"{entry.prob:.4f}", style=style)

        self.activations_table.clear(rows=True)
        if self.activation_view != "none":
            for summary in self._filtered_components(res):
                val = self._activation_display(summary)
                self.activations_table.add_row(summary.label, val)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:  # type: ignore[override]
        if event.data_table.id == "tokens_table":
            self.selected_index = int(event.row_key)
            self._refresh_detail()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:  # type: ignore[override]
        checked = event.value
        comp = event.checkbox.id
        if checked:
            self.selected_components.add(comp)
        else:
            self.selected_components.discard(comp)
        self._refresh_detail()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:  # type: ignore[override]
        if event.pressed.id:
            self.activation_view = event.pressed.id
        self._refresh_detail()

    def on_key(self, event: events.Key) -> None:
        if event.key == "j" and self.selected_index < len(self.results) - 1:
            self.selected_index += 1
            self.token_table.cursor_coordinate = (self.selected_index, 0)
            self._refresh_detail()
        elif event.key == "k" and self.selected_index > 0:
            self.selected_index -= 1
            self.token_table.cursor_coordinate = (self.selected_index, 0)
            self._refresh_detail()


################################################################################
# entrypoint
################################################################################


def main():
    args = parse_args()
    results, _ = _gather_predictions(args)
    app = ColorizeViewer(results)
    app.run()


if __name__ == "__main__":
    main()
