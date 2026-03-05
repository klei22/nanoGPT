#!/usr/bin/env python3
"""view_hp_log.py — Textual TUI that tail-views *sweep_log.yaml*

Features
========
* **Live refresh** – polls the YAML every 5 s.
* **Iterations list** – pick a number for metrics + candidate table.
* **Summary** – shows a compact table of the best config after every iteration, starting with the *iter −1* baseline row.

Keys
----
↑ / ↓   select in sidebar  q   quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
import re

import yaml
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, DataTable, Static

# ── constants ─────────────────────────────────────────────
DEFAULT_LOG = "sweep_log.yaml"
POLL_INTERVAL = 5.0  # seconds
SUMMARY_LABEL = "Summary"
HILITE_STYLE = "bold orange3"

# ── helper functions ─────────────────────────────────────


def load_yaml(path: Path) -> Dict[str, Any]:
    """Return parsed YAML or empty dict if file missing/empty."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def fnum(val: Any, spec: str) -> str:
    return spec.format(val) if isinstance(val, (int, float)) else str(val)


def metrics_panel(
        current_iter_baseline_metrics: Dict[str, Any],
        prev_iter_baseline_loss: Any = "-",       # Baseline loss from prev iter
        prev_iter_baseline_rankme: Any = "-",
        prev_iter_baseline_areq: Any = "-",
        current_avg_candidate_loss: Any = "-",    # Avg Cand Loss for current iter
        prior_avg_candidate_loss: Any = "-",      # Avg Cand Loss for prior iter
        delta_avg_candidate_loss: Any = "-",      # Difference in Avg Cand Loss
        ) -> Panel:
    g = Table.grid(padding=1)
    g.add_column(justify="right")
    g.add_column()
    g.add_row("Prior Loss", fnum(prev_iter_baseline_loss, "{:.4f}"))
    g.add_row("After Iteration Loss", fnum(current_iter_baseline_metrics.get("loss", current_iter_baseline_metrics.get("best_val_loss", "-")), "{:.4f}"))
    g.add_row("Best iter", str(current_iter_baseline_metrics.get("best_iter", "-")))

    # Average loss from all candidates explored in the current iteration
    g.add_row("Avg Cand Loss", fnum(current_avg_candidate_loss, "{:.4f}"))
    g.add_row("Prior Avg Loss", fnum(prior_avg_candidate_loss, "{:.4f}"))
    g.add_row("Δ Avg Loss", fnum(delta_avg_candidate_loss, "{:.4f}"))
    cur_rankme = current_iter_baseline_metrics.get("rankme", "-")
    cur_areq = current_iter_baseline_metrics.get("areq", "-")
    d_rankme = (cur_rankme - prev_iter_baseline_rankme) if isinstance(cur_rankme, (int, float)) and isinstance(prev_iter_baseline_rankme, (int, float)) else "-"
    d_areq = (cur_areq - prev_iter_baseline_areq) if isinstance(cur_areq, (int, float)) and isinstance(prev_iter_baseline_areq, (int, float)) else "-"
    g.add_row("Prior RankMe", fnum(prev_iter_baseline_rankme, "{:.4f}"))
    g.add_row("RankMe", fnum(cur_rankme, "{:.4f}"))
    g.add_row("Δ RankMe", fnum(d_rankme, "{:+.4f}"))
    g.add_row("Prior AReQ", fnum(prev_iter_baseline_areq, "{:.4f}"))
    g.add_row("AReQ", fnum(cur_areq, "{:.4f}"))
    g.add_row("Δ AReQ", fnum(d_areq, "{:+.4f}"))

    # This round statistics
    g.add_row("Score", fnum(current_iter_baseline_metrics.get("score", "-"), "{:.4e}")) # Baseline score
    g.add_row("Params", fnum(current_iter_baseline_metrics.get("num_params", current_iter_baseline_metrics.get("params", "-")), "{:.3e}"))

    return Panel(g, title="Iteration stats", border_style="green")


# ── TUI application ──────────────────────────────────────


class SweepViewer(App):
    CSS = """#navbox{width:16;} #main{width:1fr;}"""
    BINDINGS = [Binding("q", "quit", show=False)]

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self._mtime: float = 0.0
        self.iters: List[Dict[str, Any]] = []
        self.idx: int = 0
        self.show_summary: bool = False
        self.base_cfg: Dict[str, Any] = {}
        self.base_metrics: Dict[str, Any] = {}
        self.base_iter: Dict[str, Any] | None = None

    # ── compose UI ──────────────────────────
    def compose(self):  # type: ignore[override]
        yield Header()
        with Horizontal():
            # sidebar
            with Vertical(id="navbox"):
                yield Static("Iterations", classes="title")
                yield DataTable(id="nav", show_header=False, zebra_stripes=True)
            # main pane
            with Vertical(id="main"):
                yield Static(id="panel")
                yield DataTable(id="table", zebra_stripes=True)
        yield Footer()

    # ── mount & poll ────────────────────────
    def on_mount(self):  # type: ignore[override]
        self._load_yaml(initial=True)
        self._build_nav()
        self._refresh_view()
        self.set_interval(POLL_INTERVAL, self._poll_yaml)

    def _poll_yaml(self):
        mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0
        if mtime == self._mtime:
            return
        remember = None
        if self.iters and not self.show_summary:
            remember = self.iters[self.idx]["iter"]
        self._load_yaml()
        self._build_nav()
        if remember is not None:
            for i, it in enumerate(self.iters):
                if it["iter"] == remember:
                    self.idx = i
                    break
        self._refresh_view()

    # ── YAML load ───────────────────────────
    def _load_yaml(self, *, initial=False):
        data = load_yaml(self.log_path)
        self.base_cfg = data.get("baseline_config", {})
        self.base_metrics = data.get("baseline_metrics", {})
        self.base_iter = None
        for it in data.get("iterations", []):
            if it.get("iter") == -1:
                self.base_iter = it
                self.base_metrics = it.get("baseline_metrics", self.base_metrics)
                break
        self.iters = [
            it
            for it in data.get("iterations", [])
            if it.get("iter", 0) >= 0 and "candidates" in it
        ]
        if self.iters:
            self.idx = (
                len(self.iters) - 1 if initial else min(self.idx, len(self.iters) - 1)
            )
        else:
            self.idx = 0
        self._mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0

    # ── navigation list ─────────────────────
    def _build_nav(self):
        nav = self.query_one("#nav", DataTable)
        nav.clear(columns=True)
        nav.add_column("item")
        for it in self.iters:
            nav.add_row(str(it["iter"]))
        nav.add_row(SUMMARY_LABEL)
        nav.cursor_type = "row"
        row_coord = (
            self.idx if (self.iters and not self.show_summary) else len(self.iters),
            0,
        )
        nav.cursor_coordinate = row_coord
        nav.focus()

    def on_data_table_row_highlighted(self, e: DataTable.RowHighlighted):  # type: ignore[override]
        if e.data_table.id != "nav":
            return
        row = e.cursor_row  # type: ignore[attr-defined]
        self.show_summary = row == len(self.iters)
        if not self.show_summary and self.iters:
            self.idx = row
        self._refresh_view()

    # ── calculation helpers ───────────────
    def _calculate_avg_cand_loss(self, blk: Dict[str, Any]) -> float | None:
        """Calculates the average best_val_loss from candidates in an iteration block."""
        total_loss = 0.0
        loss_count = 0
        for candidate in blk.get("candidates", []):
            loss = candidate.get("best_val_loss")
            if isinstance(loss, (int, float)):
                total_loss += loss
                loss_count += 1
        if loss_count > 0:
            return total_loss / loss_count
        return None # Return None if no valid candidates/losses

    # ── summary helpers ─────────────────────
    def _summary_data(self):
        if not self.iters:
            return ["iter"], [["-1"]]
        # guard against iterations where “chosen” is missing/None
        changed = sorted({it["chosen"]["param"] for it in self.iters if it.get("chosen")})
        hdrs = [
            "iter",
            *changed,
            "best_loss",
            "best_iter",
            "rankme",
            "Δrankme",
            "areq",
            "Δareq",
            "params",
            "Δparams",
            "eff.",
        ]

        # helper ────────────────────────────────────────────────────────────
        def _lookup(cfg: Dict[str, Any], key: str) -> Any:
            """
            Return cfg[key] unless key looks like  ‘something_layerlist[N]’.
            In that case dig into the list and fetch element N (if present),
            otherwise return “-”.
            """
            m = re.fullmatch(r"(\w+_layerlist)\[(\d+)]", key)
            if not m:
                return cfg.get(key, "-")
            list_key, idx_s = m.groups()
            idx = int(idx_s)
            lst = cfg.get(list_key)
            if isinstance(lst, list) and idx < len(lst):
                return lst[idx]
            return "-"  # list missing or too short
        rows: List[List[Any]] = []

        # baseline (iter -1) row
        base_src = (self.base_iter or {}).get("baseline_config_after", self.base_cfg)
        base_vals = [str(_lookup(base_src, p)) for p in changed]
        rows.append(
            [
                "-1",
                *base_vals,
                fnum(self.base_metrics.get("loss", "-"), "{:.4f}"),
                str(self.base_metrics.get("best_iter", "-")),
                fnum(self.base_metrics.get("rankme", "-"), "{:.4f}"),
                "-",
                fnum(self.base_metrics.get("areq", "-"), "{:.4f}"),
                "-",
                fnum(self.base_metrics.get("params", "-"), "{:,}"),
                "-",
                "-",
            ]
        )
        for i, it in enumerate(self.iters):
            ch = it.get("chosen") or {}                    # could be None
            after = it["baseline_config_after"]

            # highlight the changed field only if “chosen” is present
            vals = []
            for p in changed:
                val = _lookup(after, p)
                if ch and p == ch.get("param"):
                    vals.append(Text(str(val), style=HILITE_STYLE))
                else:
                    vals.append(str(val))

            # stats columns – fall back to “-” if missing
            if ch:
                vals += [
                    f"{ch['best_val_loss']:.4f}",
                    str(ch.get("best_iter", "-")),
                    fnum(ch.get("avg_rankme", "-"), "{:.4f}"),
                    fnum(ch.get("delta_rankme", "-"), "{:+.4f}"),
                    fnum(ch.get("avg_areq", "-"), "{:.4f}"),
                    fnum(ch.get("delta_areq", "-"), "{:+.4f}"),
                    f"{int(ch['num_params']):,}",
                    f"{int(ch['delta_params']):,}",
                    f"{ch['efficiency']:.2e}",
                ]
            else:
                vals += ["-", "-", "-", "-", "-", "-", "-", "-", "-"]
            rows.append([str(i), *vals])
        return hdrs, rows

    # ── UI refresh ─────────────────────────
    def _refresh_view(self):
        panel = self.query_one("#panel", Static)
        table = self.query_one("#table", DataTable)

        # waiting for data …
        if not self.iters:
            table.visible = False
            panel.update(Panel("Waiting for data… (polling)", border_style="red"))
            self.sub_title = "No data yet"
            return

        table.visible = True
        if self.show_summary:
            panel.update(
                Panel("Summary (best config per iteration)", border_style="cyan")
            )
            hdrs, rows = self._summary_data()
            table.clear(columns=True)
            table.add_columns(*hdrs)
            for r in rows:
                table.add_row(*[c if isinstance(c, Text) else str(c) for c in r])
            self.sub_title = "Summary view"
            return

        blk = self.iters[self.idx]
        current_baseline_metrics = blk["baseline_metrics"]

        prior_loss_val = "-"
        prior_rankme_val = "-"
        prior_areq_val = "-"
        if self.idx == 0: # Current iteration is the first one (e.g., iter 0)
            if self.base_metrics: # Use metrics from iter -1 or initial baseline
                prior_loss_val = self.base_metrics.get("loss", self.base_metrics.get("best_val_loss", "-"))
                prior_rankme_val = self.base_metrics.get("rankme", "-")
                prior_areq_val = self.base_metrics.get("areq", "-")
        elif self.idx > 0: # Current iteration is not the first one
            # Use baseline_metrics from the previous iteration in the list
            prev_iter_baseline_metrics = self.iters[self.idx - 1]["baseline_metrics"]
            prior_loss_val = prev_iter_baseline_metrics.get("loss", prev_iter_baseline_metrics.get("best_val_loss", "-"))
            prior_rankme_val = prev_iter_baseline_metrics.get("rankme", "-")
            prior_areq_val = prev_iter_baseline_metrics.get("areq", "-")


        # Calculate Avg Cand Loss for current and prior iterations
        current_avg_loss_val = self._calculate_avg_cand_loss(blk) # Returns float or None

        prior_avg_loss_val = None
        if self.idx > 0:
            prior_blk = self.iters[self.idx - 1]
            prior_avg_loss_val = self._calculate_avg_cand_loss(prior_blk) # Returns float or None

        # Calculate Delta Avg Cand Loss
        delta_avg_loss_val = None
        if isinstance(current_avg_loss_val, float) and isinstance(prior_avg_loss_val, float):
            delta_avg_loss_val = current_avg_loss_val - prior_avg_loss_val

        # Format for display
        current_avg_loss_display = f"{current_avg_loss_val:.4f}" if current_avg_loss_val is not None else "-"
        prior_avg_loss_display = f"{prior_avg_loss_val:.4f}" if prior_avg_loss_val is not None else "-"
        delta_avg_loss_display = f"{delta_avg_loss_val:.4f}" if delta_avg_loss_val is not None else "-"

        panel.update(metrics_panel(
            current_iter_baseline_metrics=current_baseline_metrics,
            prev_iter_baseline_loss=prior_loss_val,
            current_avg_candidate_loss=current_avg_loss_display,
            prior_avg_candidate_loss=prior_avg_loss_display,
            delta_avg_candidate_loss=delta_avg_loss_display,
            prev_iter_baseline_rankme=prior_rankme_val,
            prev_iter_baseline_areq=prior_areq_val,
        ))
        self.sub_title = f"Iteration {blk['iter']}  (↑/↓ nav, q quit)"
        table.clear(columns=True)
        table.add_columns(
            "param", "value", "best_loss", "best_iter", "rankme", "Δrankme", "areq", "Δareq", "Δscore", "Δparams", "eff."
        )
        # “chosen” may be absent (e.g., no best candidate selected this round)
        chosen = blk.get("chosen") or {}
        for c in blk["candidates"]:
            hl = (
                bool(chosen)
                and c["param"] == chosen.get("param")
                and c["value"] == chosen.get("value")
            )
            st = "bold yellow" if hl else ""
            table.add_row(
                Text(str(c["param"]), style=st),
                Text(str(c["value"]), style=st),
                f"{c['best_val_loss']:.4f}",
                str(c.get("best_iter", "-")),
                fnum(c.get("avg_rankme", "-"), "{:.4f}"),
                fnum(c.get("delta_rankme", "-"), "{:+.4f}"),
                fnum(c.get("avg_areq", "-"), "{:.4f}"),
                fnum(c.get("delta_areq", "-"), "{:+.4f}"),
                f"{c['delta_score']:.2e}",
                f"{c['delta_params']:.2e}",
                f"{c['efficiency']:.2e}",
            )


# ── entry point ─────────────────────────────


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_LOG)
    SweepViewer(path).run()


if __name__ == "__main__":
    main()
