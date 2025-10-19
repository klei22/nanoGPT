# explore_trained_model_with_dataset.py
"""Interactive dataset explorer for trained GPT checkpoints.

This script is a refactor of the old ``colorize_dataset.py`` tool.  It keeps
all of the original command line features for loading a checkpoint, stepping
through a dataset split, and computing the same statistics/metrics.  Instead of
printing to stdout it now launches a Textual UI with three live panels:

* **Stats panel** – cross entropy, ranks, probabilities, etc. for the selected
  token.
* **Activation panel** – per-layer values (``wte``, ``attn``, ``mlp``,
  ``resid``) rendered in columns similar to ``run_exploration_monitor.py``.
* **Top‑k panel** – probabilities/logits for the top predictions.

Keyboard shortcuts allow switching views (activation mode, token colouring
mode, toggling the colour strip) and exporting the collected metrics to CSV.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, Static
import tiktoken  # type: ignore

from model import GPT, GPTConfig


################################################################################
# dataclasses / helpers
################################################################################


@dataclass
class LayerInfo:
    """Container describing layer-aligned statistics for the selected token."""

    label: str
    dot_target: float
    best_token_id: int
    best_token_text: str
    rank: int


@dataclass
class TopKEntry:
    token_id: int
    token: str
    logit: float
    prob: float
    rank: int
    is_target: bool


@dataclass
class TokenRecord:
    index: int
    dataset_pos: int
    context_length: int
    target_id: int
    target_text: str
    cross_entropy: float
    focal: float
    prob_target: float
    prob_left: float
    prob_top1: float
    rank: int
    logit_target: float
    topk: List[TopKEntry]
    layers: List[LayerInfo]


def _ansi(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor").print(renderable)
    return buf.getvalue()


def _escape_ws(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _colour(
    ids: Sequence[int],
    scalars: Sequence[float],
    decode: Callable[[Sequence[int]], str],
    *,
    escape_ws: bool,
) -> Text:
    vals = torch.tensor(scalars, dtype=torch.float32)
    if len(vals) == 0:
        return Text("(no tokens)")
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


################################################################################
# byte-fallback helpers -------------------------------------------------------
################################################################################


def _ccwb_encode(text: str, stoi):
    lst: List[int] = []
    for ch in text:
        if ch in stoi:
            lst.append(stoi[ch])
        else:
            lst.extend(stoi[bytes([b])] for b in ch.encode())
    return lst


def _ccwb_decode(ids: Sequence[int], itos):
    out: List[str] = []
    buf: List[bytes] = []

    def flush() -> None:
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


################################################################################
# CLI / loading
################################################################################


def parse_args():
    p = argparse.ArgumentParser("Explore a dataset split with a trained GPT model")
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
    p.add_argument("--csv_file", default="dataset_metrics.csv", help="Default CSV path for exports")
    p.add_argument("--display", choices=["token", "topk"], default="token", help="Initial display mode")
    p.add_argument("--topk", type=int, default=10, help="Number of top predictions to display")
    p.add_argument("--max_token_chars", type=int, default=20, help="Maximum characters for top-k token columns (-1 to disable clipping)")
    p.add_argument("--rank_red", type=int, default=100, help="Rank value treated as fully red in heatmap")
    p.add_argument("--target_style", default="cyan", help="Colour to highlight the target token or 'underline' to underline it")
    p.add_argument("--bold_target", action=argparse.BooleanOptionalAction, default=True, help="Bold the target token when highlighting")
    p.add_argument("--escape_whitespace", action=argparse.BooleanOptionalAction, default=True, help="Show newline and tab characters as escape sequences")
    p.add_argument("--plot_metrics", action="store_true", help="Generate Plotly graphs for prediction metrics")
    p.add_argument("--plot_file", default="metrics.html", help="Output HTML file for Plotly metrics")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma for metrics plotting")
    p.add_argument(
        "--activation_view",
        choices=["target", "rank", "rank_word", "none"],
        default="target",
        help=(
            "Per-layer activation columns: 'target' shows dot-product with target token, "
            "'rank' shows rank of token best aligned with activation, 'rank_word' also "
            "shows that token alongside its rank, and 'none' hides these columns"
        ),
    )
    p.add_argument(
        "--components",
        choices=["wte", "attn", "mlp", "resid"],
        nargs="+",
        default=["wte", "attn", "mlp"],
        help="Activation components to collect in activation_view modes",
    )
    return p.parse_args()


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


################################################################################
# Textual helpers / screens
################################################################################


class FileNameScreen(Screen[str | None]):
    """Modal prompt returning a CSV filename or ``None`` when cancelled."""

    CSS = """
    Screen { align: center middle; }
    #prompt { margin-bottom: 1; }
    """

    def compose(self) -> ComposeResult:  # noqa: D401
        yield Label("Enter CSV filename:", id="prompt")
        yield Input(placeholder="results.csv", id="fname")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)


class DatasetExplorerApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #token-strip {
        height: auto;
        padding: 1 2;
        border: round $surface;
        margin: 0 1;
    }
    #main-container {
        layout: horizontal;
        height: 1fr;
        padding: 1;
        gap: 1;
    }
    DataTable {
        width: 1fr;
        height: 1fr;
        border: round $secondary;
    }
    """

    BINDINGS = [
        Binding("left", "prev_token", "Previous token"),
        Binding("right", "next_token", "Next token"),
        Binding("m", "toggle_mode", "Toggle colour mode"),
        Binding("v", "cycle_activation_view", "Cycle activation view"),
        Binding("t", "toggle_token_strip", "Toggle colour strip"),
        Binding("e", "export_csv", "Export CSV"),
        Binding("E", "export_csv_prompt", "Export CSV (prompt)"),
        Binding("?", "show_help", "Show help"),
        Binding("q", "quit", "Quit"),
    ]

    activation_view = reactive("target")
    mode = reactive("minmax")
    show_strip = reactive(True)
    index = reactive(0)

    def __init__(self, args, records: List[TokenRecord], colour_texts: dict[str, Text]):
        super().__init__()
        self.args = args
        self.records = records
        self.colour_texts = colour_texts
        self.activation_view = args.activation_view
        self.mode = args.mode
        self.show_strip = args.display == "token"

    # ------------------------------------------------------------------ compose
    def compose(self) -> ComposeResult:  # noqa: D401
        yield Header(show_clock=True)
        yield Static(id="token-strip")
        with Container(id="main-container"):
            yield DataTable(id="stats")
            yield DataTable(id="activations")
            yield DataTable(id="topk")
        yield Footer()

    # ------------------------------------------------------------------ helpers
    @property
    def current_record(self) -> TokenRecord:
        return self.records[self.index]

    # ------------------------------------------------------------------ mount
    def on_mount(self) -> None:
        if not self.records:
            self.bell()
            self.exit(message="No tokens to display")
            return
        self.refresh_all()

    # ------------------------------------------------------------------ watchers
    def watch_mode(self, _: str) -> None:
        self.refresh_strip()
        self.refresh_title()

    def watch_activation_view(self, _: str) -> None:
        self.refresh_activations()
        self.refresh_title()

    def watch_show_strip(self, _: bool) -> None:
        self.refresh_strip()

    def watch_index(self, _: int) -> None:
        self.refresh_all()

    # ------------------------------------------------------------------ refresh
    def refresh_all(self) -> None:
        self.index = max(0, min(self.index, len(self.records) - 1))
        self.refresh_strip()
        self.refresh_stats()
        self.refresh_activations()
        self.refresh_topk()
        self.refresh_title()

    def refresh_strip(self) -> None:
        widget = self.query_one("#token-strip", Static)
        if not self.show_strip:
            widget.display = False
            return
        widget.display = True
        widget.update(self.colour_texts.get(self.mode, Text("")))

    def refresh_stats(self) -> None:
        table = self.query_one("#stats", DataTable)
        if not table.columns:
            table.add_column("Metric", width=24)
            table.add_column("Value")
        else:
            table.clear()

        rec = self.current_record
        rows = [
            ("Token index", str(rec.index)),
            ("Dataset position", str(rec.dataset_pos)),
            ("Context length", str(rec.context_length)),
            ("Target token", rec.target_text),
            ("Target id", str(rec.target_id)),
            ("Rank", str(rec.rank)),
            ("Prob(target)", f"{rec.prob_target:.6f}"),
            ("Prob(left)", f"{rec.prob_left:.6f}"),
            ("Prob(top1)", f"{rec.prob_top1:.6f}"),
            ("Cross entropy", f"{rec.cross_entropy:.6f}"),
            ("Focal", f"{rec.focal:.6f}"),
        ]
        for metric, value in rows:
            table.add_row(metric, value)

    def refresh_activations(self) -> None:
        table = self.query_one("#activations", DataTable)
        table.clear(columns=True)
        rec = self.current_record
        if not rec.layers or self.activation_view == "none":
            table.add_column("Info")
            table.add_row("Activation view disabled")
            return

        labels = [layer.label for layer in rec.layers]
        for label in labels:
            table.add_column(label, no_wrap=True)

        if self.activation_view == "target":
            values = [layer.dot_target for layer in rec.layers]
            t = torch.tensor(values)
            norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
            row = []
            for value, nv in zip(values, norm.tolist()):
                r = int((1 - nv) * 255)
                g = int(nv * 255)
                row.append(Text(f"{value:.2f}", style=f"bold #{r:02x}{g:02x}00"))
            table.add_row(*row)
        elif self.activation_view == "rank":
            row = []
            for layer in rec.layers:
                rank_norm = 1 - (min(layer.rank, self.args.rank_red) - 1) / max(self.args.rank_red - 1, 1)
                r = int((1 - rank_norm) * 255)
                g = int(rank_norm * 255)
                row.append(Text(str(layer.rank), style=f"bold #{r:02x}{g:02x}00"))
            table.add_row(*row)
        else:  # rank_word
            row = []
            for layer in rec.layers:
                rank_norm = 1 - (min(layer.rank, self.args.rank_red) - 1) / max(self.args.rank_red - 1, 1)
                r = int((1 - rank_norm) * 255)
                g = int(rank_norm * 255)
                row.append(Text(f"{layer.best_token_text}:{layer.rank}", style=f"bold #{r:02x}{g:02x}00"))
            table.add_row(*row)

    def refresh_topk(self) -> None:
        table = self.query_one("#topk", DataTable)
        table.clear(columns=True)
        table.add_column("Token", no_wrap=True)
        table.add_column("Prob", justify="right")
        table.add_column("Logit", justify="right")
        table.add_column("Rank", justify="right")

        rec = self.current_record
        logits = torch.tensor([entry.logit for entry in rec.topk], dtype=torch.float32)
        norm = (logits - logits.min()) / (logits.max() - logits.min() + 1e-6)
        for entry, nv in zip(rec.topk, norm.tolist()):
            r = int((1 - nv) * 255)
            g = int(nv * 255)
            style = f"#{r:02x}{g:02x}00"
            if entry.is_target:
                if self.args.target_style == "underline":
                    style = "underline"
                else:
                    style = self.args.target_style
                if self.args.bold_target:
                    style = f"bold {style}" if style else "bold"
            table.add_row(
                Text(entry.token, style=style),
                Text(f"{entry.prob:.6f}", style=f"#{r:02x}{g:02x}00"),
                Text(f"{entry.logit:.3f}", style=f"#{r:02x}{g:02x}00"),
                Text(str(entry.rank)),
            )

    def refresh_title(self) -> None:
        rec = self.current_record
        self.title = (
            f"Token {rec.index + 1}/{len(self.records)} – "
            f"mode: {self.mode} – activation: {self.activation_view}"
        )

    # ------------------------------------------------------------------ actions
    def action_prev_token(self) -> None:
        if self.index > 0:
            self.index -= 1

    def action_next_token(self) -> None:
        if self.index < len(self.records) - 1:
            self.index += 1

    def action_toggle_mode(self) -> None:
        self.mode = "softmax" if self.mode == "minmax" else "minmax"

    def action_cycle_activation_view(self) -> None:
        order = ["target", "rank", "rank_word", "none"]
        nxt = (order.index(self.activation_view) + 1) % len(order)
        self.activation_view = order[nxt]

    def action_toggle_token_strip(self) -> None:
        self.show_strip = not self.show_strip

    def action_export_csv(self) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(self.args.csv_file)
        if path.is_dir():
            path = path / f"dataset_metrics_{ts}.csv"
        elif path.exists():
            stem = path.stem
            path = path.with_name(f"{stem}_{ts}{path.suffix}")
        self._write_csv(path)
        self.toast(f"Saved CSV → {path}")

    def action_export_csv_prompt(self) -> None:
        self.push_screen(FileNameScreen(), self._prompt_callback)

    def _prompt_callback(self, filename: str | None) -> None:
        if not filename:
            return
        path = Path(filename).expanduser()
        if path.is_dir():
            path = path / "dataset_metrics.csv"
        self._write_csv(path)
        self.toast(f"Saved CSV → {path}")

    def _write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "index",
                "dataset_pos",
                "context_length",
                "target_id",
                "target_text",
                "rank",
                "prob_target",
                "prob_left",
                "prob_top1",
                "cross_entropy",
                "focal",
            ]
            for layer in self.records[0].layers:
                header.extend([
                    f"{layer.label}_dot_target",
                    f"{layer.label}_rank",
                    f"{layer.label}_best_token",
                ])
            writer.writerow(header)
            for rec in self.records:
                row = [
                    rec.index,
                    rec.dataset_pos,
                    rec.context_length,
                    rec.target_id,
                    rec.target_text,
                    rec.rank,
                    rec.prob_target,
                    rec.prob_left,
                    rec.prob_top1,
                    rec.cross_entropy,
                    rec.focal,
                ]
                for layer in rec.layers:
                    row.extend([layer.dot_target, layer.rank, layer.best_token_text])
                writer.writerow(row)

    def action_show_help(self) -> None:
        self.toast(
            "←/→: tokens | m: mode | v: activation view | t: toggle strip | e/E: export CSV | q: quit",
            timeout=6,
        )


################################################################################
# computation
################################################################################


def collect_records(args) -> tuple[List[TokenRecord], dict[str, Text], dict[str, list[float]]]:
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

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
        if "cuda" in args.device
        else torch.no_grad()
    )

    block = args.block_size or model.config.block_size
    pos = args.offset
    tokens_left = min(args.num_tokens, len(data) - 1 - pos)

    records: List[TokenRecord] = []
    colour_lists = {"minmax": [], "softmax": []}

    metrics: dict[str, list[float]] = {
        "rank": [],
        "prob_left": [],
        "prob_tgt": [],
        "prob_top1": [],
        "cross_entropy": [],
        "focal": [],
    }

    index = 0
    while tokens_left > 0:
        seq = data[pos : pos + block + 1]
        if len(seq) < 2:
            break

        ctx_tok = torch.from_numpy(seq[:-1].astype(np.int64))[None].to(args.device)

        activations = {"t0": None, "attn": [], "mlp": [], "ar": [], "mr": []}
        handles = []

        if args.activation_view != "none" or args.components:

            def t0_hook(module, inp, out):
                activations["t0"] = out[0, -1, :].detach()

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

        if "resid" in args.components and activations["t0"] is not None:
            resid = activations["t0"].float().clone()
            activations["ar"].clear()
            activations["mr"].clear()
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
        tgt_logit = logits[-1, tgt_token].item()
        rank = int((logits[-1] > logits[-1, tgt_token]).sum().item()) + 1
        prob_left = probs[logits[-1] > logits[-1, tgt_token]].sum().item()
        p_top1 = probs.max().item()
        ce = -math.log(tgt_prob + 1e-12)
        focal = ((1 - tgt_prob) ** args.focal_gamma) * ce

        metrics["rank"].append(rank)
        metrics["prob_left"].append(prob_left)
        metrics["prob_tgt"].append(tgt_prob)
        metrics["prob_top1"].append(p_top1)
        metrics["cross_entropy"].append(ce)
        metrics["focal"].append(focal)

        layer_infos: List[LayerInfo] = []
        if activations["t0"] is not None and args.components:
            vecs: List[tuple[str, torch.Tensor]] = []
            if "wte" in args.components and activations["t0"] is not None:
                vecs.append(("t0", activations["t0"].float()))
            for i in range(model.config.n_layer):
                if "attn" in args.components and i < len(activations["attn"]):
                    vecs.append((f"a{i + 1}", activations["attn"][i].float()))
                if "resid" in args.components and i < len(activations["ar"]):
                    vecs.append((f"ar{i + 1}", activations["ar"][i].float()))
                if "mlp" in args.components and i < len(activations["mlp"]):
                    vecs.append((f"m{i + 1}", activations["mlp"][i].float()))
                if "resid" in args.components and i < len(activations["mr"]):
                    vecs.append((f"mr{i + 1}", activations["mr"][i].float()))

            emb = model.lm_head.weight.detach().float()
            correct_vec = emb[tgt_token].clone()
            for label, vec in vecs:
                dot_target = torch.dot(vec, correct_vec).item()
                logits_vec = emb @ vec
                best_token_id = int(torch.argmax(logits_vec).item())
                layer_rank = int((logits[-1] > logits[-1, best_token_id]).sum().item()) + 1
                best_token = decode([best_token_id])
                if args.max_token_chars >= 0:
                    best_token = best_token[: args.max_token_chars]
                if args.escape_whitespace:
                    best_token = _escape_ws(best_token)
                layer_infos.append(
                    LayerInfo(
                        label=label,
                        dot_target=dot_target,
                        best_token_id=best_token_id,
                        best_token_text=best_token,
                        rank=layer_rank,
                    )
                )

        topv, topi = logits[-1].topk(args.topk)
        top_entries: List[TopKEntry] = []
        for idx, logit_val in zip(topi.tolist(), topv.tolist()):
            token = decode([idx])
            if args.max_token_chars >= 0:
                token = token[: args.max_token_chars]
            if args.escape_whitespace:
                token = _escape_ws(token)
            entry_rank = int((logits[-1] > logits[-1, idx]).sum().item()) + 1
            top_entries.append(
                TopKEntry(
                    token_id=idx,
                    token=token,
                    logit=logit_val,
                    prob=probs[idx].item(),
                    rank=entry_rank,
                    is_target=idx == tgt_token,
                )
            )

        target_word = decode([tgt_token])
        if args.max_token_chars >= 0:
            target_word = target_word[: args.max_token_chars]
        if args.escape_whitespace:
            target_word = _escape_ws(target_word)

        records.append(
            TokenRecord(
                index=index,
                dataset_pos=pos,
                context_length=ctx_len,
                target_id=tgt_token,
                target_text=target_word,
                cross_entropy=ce,
                focal=focal,
                prob_target=tgt_prob,
                prob_left=prob_left,
                prob_top1=p_top1,
                rank=rank,
                logit_target=tgt_logit,
                topk=top_entries,
                layers=layer_infos,
            )
        )

        colour_lists["minmax"].append(tgt_logit)
        colour_lists["softmax"].append(tgt_prob)

        step = 1 if args.window == "rolling" else ctx_len
        pos += step
        tokens_left -= 1 if args.window == "rolling" else min(ctx_len, tokens_left)
        index += 1

    ids = [rec.target_id for rec in records]
    colour_texts = {
        key: _colour(ids, values, decode, escape_ws=args.escape_whitespace)
        for key, values in colour_lists.items()
    }

    if args.output_file:
        if args.display == "token":
            render = colour_texts[args.mode]
            Path(args.output_file).write_text(_ansi(render), "utf-8", errors="replace")
            console.print(f"[cyan]Saved colourised tokens → {args.output_file}[/cyan]")
        else:
            table = build_rich_table(records, args)
            Path(args.output_file).write_text(_ansi(table), "utf-8", errors="replace")
            console.print(f"[cyan]Saved table → {args.output_file}[/cyan]")

    return records, colour_texts, metrics


################################################################################
# rich table builder (for optional file output)
################################################################################


def build_rich_table(records: List[TokenRecord], args) -> Table:
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("target", no_wrap=True)
    table.add_column("xent", justify="right", no_wrap=True)
    table.add_column("rank", justify="right", no_wrap=True)
    table.add_column("p_tgt", justify="right", no_wrap=True)
    table.add_column("p_left", justify="right", no_wrap=True)

    if records and records[0].layers and args.activation_view != "none":
        for layer in records[0].layers:
            table.add_column(layer.label, justify="right", no_wrap=True)

    for i in range(args.topk):
        table.add_column(f"top{i + 1}", justify="center", no_wrap=True)

    for rec in records:
        layer_cells: List[Text] = []
        if rec.layers and args.activation_view != "none":
            if args.activation_view == "target":
                vals = [layer.dot_target for layer in rec.layers]
                t = torch.tensor(vals)
                norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
                for val, nv in zip(vals, norm.tolist()):
                    r = int((1 - nv) * 255)
                    g = int(nv * 255)
                    layer_cells.append(Text(f"{val:.2f}", style=f"bold #{r:02x}{g:02x}00"))
            elif args.activation_view == "rank":
                for layer in rec.layers:
                    rank_norm = 1 - (min(layer.rank, args.rank_red) - 1) / max(args.rank_red - 1, 1)
                    r = int((1 - rank_norm) * 255)
                    g = int(rank_norm * 255)
                    layer_cells.append(Text(str(layer.rank), style=f"bold #{r:02x}{g:02x}00"))
            else:
                for layer in rec.layers:
                    rank_norm = 1 - (min(layer.rank, args.rank_red) - 1) / max(args.rank_red - 1, 1)
                    r = int((1 - rank_norm) * 255)
                    g = int(rank_norm * 255)
                    layer_cells.append(Text(f"{layer.best_token_text}:{layer.rank}", style=f"bold #{r:02x}{g:02x}00"))

        topk_cells: List[Text] = []
        logits = torch.tensor([entry.logit for entry in rec.topk], dtype=torch.float32)
        norm = (logits - logits.min()) / (logits.max() - logits.min() + 1e-6)
        for entry, nv in zip(rec.topk, norm.tolist()):
            r = int((1 - nv) * 255)
            g = int(nv * 255)
            style = f"#{r:02x}{g:02x}00"
            if entry.is_target:
                if args.target_style == "underline":
                    style = "underline"
                else:
                    style = args.target_style
                if args.bold_target:
                    style = f"bold {style}" if style else "bold"
            topk_cells.append(Text(entry.token, style=style))

        rank_norm = 1 - (min(rec.rank, args.rank_red) - 1) / max(args.rank_red - 1, 1)
        rr = int((1 - rank_norm) * 255)
        rg = int(rank_norm * 255)
        rank_text = Text(str(rec.rank), style=f"bold #{rr:02x}{rg:02x}00")

        v = rec.prob_target
        pr = int((1 - v) * 255)
        pg = int(v * 255)
        p_tgt_text = Text(f"{rec.prob_target:.4f}", style=f"bold #{pr:02x}{pg:02x}00")

        v = 1 - rec.prob_left
        lr = int((1 - v) * 255)
        lg = int(v * 255)
        p_left_text = Text(f"{rec.prob_left:.4f}", style=f"bold #{lr:02x}{lg:02x}00")

        target_style = args.target_style if args.target_style != "underline" else "underline"
        if args.bold_target:
            target_style = f"bold {target_style}" if target_style else "bold"

        table.add_row(
            Text(rec.target_text, style=target_style),
            f"{rec.cross_entropy:.4f}",
            rank_text,
            p_tgt_text,
            p_left_text,
            *layer_cells,
            *topk_cells,
        )

    return table


################################################################################
# main
################################################################################


def main() -> None:
    args = parse_args()
    records, colour_texts, metrics = collect_records(args)

    if not records:
        Console().print("[red]No tokens gathered; exiting.[/red]")
        return

    app = DatasetExplorerApp(args, records, colour_texts)
    app.run()

    if args.plot_metrics:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        x = list(range(len(metrics["rank"])))
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
        fig.add_trace(go.Scatter(x=x, y=metrics["rank"], name="rank"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics["prob_left"], name="p_left"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics["prob_tgt"], name="p_tgt"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics["prob_top1"], name="p_top1"), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics["cross_entropy"], name="cross_entropy"), row=5, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics["focal"], name="focal"), row=6, col=1)
        fig.update_layout(height=300 * 6, showlegend=False)
        fig.write_html(args.plot_file)
        Console().print(f"[cyan]Saved plot → {args.plot_file}[/cyan]")


if __name__ == "__main__":
    main()
