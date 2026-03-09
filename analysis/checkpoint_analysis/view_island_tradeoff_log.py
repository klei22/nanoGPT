#!/usr/bin/env python3
"""TUI viewer for island tradeoff search_log.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from rich.panel import Panel
from rich.table import Table
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

DEFAULT_LOG = "search_log.yaml"
POLL_INTERVAL = 3.0
SUMMARY_LABEL = "Selected Summary"


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


class IslandTradeoffViewer(App):
    CSS = """#navbox{width:24;} #main{width:1fr;}"""
    BINDINGS = [Binding("q", "quit", show=False)]

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.rounds: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.idx = 0
        self.show_summary = False
        self._mtime = 0.0

    def compose(self):  # type: ignore[override]
        yield Header()
        with Horizontal():
            with Vertical(id="navbox"):
                yield Static("Rounds", classes="title")
                yield DataTable(id="nav", show_header=False, zebra_stripes=True)
            with Vertical(id="main"):
                yield Static(id="top")
                yield DataTable(id="table", zebra_stripes=True)
        yield Footer()

    def on_mount(self):  # type: ignore[override]
        self._reload(initial=True)
        self._build_nav()
        self._refresh()
        self.set_interval(POLL_INTERVAL, self._poll)

    def _poll(self):
        mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0
        if mtime != self._mtime:
            remember = self.idx
            self._reload(initial=False)
            self.idx = min(remember, max(0, len(self.rounds) - 1))
            self._build_nav()
            self._refresh()

    def _reload(self, initial: bool):
        data = load_yaml(self.log_path)
        self.rounds = data.get("rounds", [])
        self.summary = {
            "dataset": data.get("dataset", "-"),
            "baseline": data.get("baseline", {}),
            "search": data.get("search", {}),
            "selected": data.get("selected", {}),
        }
        if initial and self.rounds:
            self.idx = len(self.rounds) - 1
        self._mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0

    def _build_nav(self):
        nav = self.query_one("#nav", DataTable)
        nav.clear(columns=True)
        nav.add_column("item")
        for r in self.rounds:
            nav.add_row(f"round {r.get('round', '?')}")
        nav.add_row(SUMMARY_LABEL)
        nav.cursor_type = "row"
        nav.cursor_coordinate = (len(self.rounds) if self.show_summary else self.idx, 0)
        nav.focus()

    def on_data_table_row_highlighted(self, e: DataTable.RowHighlighted):  # type: ignore[override]
        if e.data_table.id != "nav":
            return
        row = e.cursor_row  # type: ignore[attr-defined]
        self.show_summary = row == len(self.rounds)
        if not self.show_summary:
            self.idx = row
        self._refresh()

    def _refresh(self):
        top = self.query_one("#top", Static)
        table = self.query_one("#table", DataTable)
        table.clear(columns=True)

        if self.show_summary:
            sel = self.summary.get("selected", {})
            base = self.summary.get("baseline", {})
            srch = self.summary.get("search", {})
            panel_tbl = Table.grid(padding=1)
            panel_tbl.add_column(justify="right")
            panel_tbl.add_column()
            panel_tbl.add_row("Dataset", str(self.summary.get("dataset", "-")))
            panel_tbl.add_row("Baseline val", f"{base.get('val_loss', '-')}")
            panel_tbl.add_row("Max allowed", f"{base.get('max_allowed_val_loss', '-')}")
            panel_tbl.add_row("Selected val", f"{sel.get('val_loss', '-')}")
            panel_tbl.add_row("Loss delta %", f"{sel.get('loss_delta_pct', '-')}")
            panel_tbl.add_row("Decode proxy", f"{sel.get('decode_reduction_proxy', '-')}")
            panel_tbl.add_row("Accepted rounds", f"{srch.get('accepted_rounds', '-')}")
            top.update(Panel(panel_tbl, title="Selected summary", border_style="green"))

            table.add_column("tensor")
            table.add_column("num_islands")
            table.add_column("threshold")
            for tname, cfg in sorted(sel.get("config", {}).items()):
                table.add_row(tname, str(cfg.get("num_islands", "-")), str(cfg.get("threshold", "-")))
            return

        if not self.rounds:
            top.update("No rounds in log yet.")
            return

        r = self.rounds[self.idx]
        panel_tbl = Table.grid(padding=1)
        panel_tbl.add_column(justify="right")
        panel_tbl.add_column()
        panel_tbl.add_row("Round", str(r.get("round", "-")))
        panel_tbl.add_row("Current val", str(r.get("current_val_loss", "-")))
        panel_tbl.add_row("Stop reason", str(r.get("stop_reason", "-")))
        best = r.get("best_candidate") or {}
        panel_tbl.add_row("Best tensor", str(best.get("tensor", "-")))
        panel_tbl.add_row("Best Δloss%", str(best.get("loss_delta_pct", "-")))
        sel = r.get("selected") or {}
        panel_tbl.add_row("Selected tensor", str(sel.get("tensor", "-")))
        panel_tbl.add_row("Selected val", str(sel.get("val_loss", "-")))
        top.update(Panel(panel_tbl, title="Round detail", border_style="cyan"))

        table.add_column("tensor")
        table.add_column("from_thr")
        table.add_column("to_thr")
        table.add_column("from_islands")
        table.add_column("to_islands")
        table.add_column("val_loss")
        table.add_column("loss_delta_pct")
        table.add_column("decode_proxy")
        table.add_column("feasible")
        for cand in r.get("tested", []):
            table.add_row(
                str(cand.get("tensor", "-")),
                str(cand.get("from_threshold", "-")),
                str(cand.get("to_threshold", "-")),
                str(cand.get("from_num_islands", "-")),
                str(cand.get("to_num_islands", "-")),
                str(cand.get("val_loss", "-")),
                str(cand.get("loss_delta_pct", "-")),
                str(cand.get("decode_reduction_proxy", "-")),
                "yes" if cand.get("feasible") else "no",
            )


def main():
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_LOG)
    IslandTradeoffViewer(path).run()


if __name__ == "__main__":
    main()
