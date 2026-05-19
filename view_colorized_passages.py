#!/usr/bin/env python3
"""TUI viewer for highlighted_passage.yaml timelines.

Use Left/Right arrows (or h/l) to move between snapshots created during
training via --colorize_val_passage.
"""

import argparse
from pathlib import Path

import yaml
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static


class PassageViewer(App):
    BINDINGS = [
        ("left", "prev", "Prev"),
        ("right", "next", "Next"),
        ("h", "prev", "Prev"),
        ("l", "next", "Next"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, records):
        super().__init__()
        self.records = records
        self.index = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="passage")
        yield Footer()

    def on_mount(self) -> None:
        self._render_current()

    def action_prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._render_current()

    def action_next(self) -> None:
        if self.index < len(self.records) - 1:
            self.index += 1
            self._render_current()

    def _render_current(self) -> None:
        record = self.records[self.index]
        ansi = record.get("ansi", "")
        passage_text = Text.from_ansi(ansi)
        title = (
            f"{self.index + 1}/{len(self.records)}  "
            f"iter={record.get('iter_num')}  dataset={record.get('dataset')}  "
            f"val_loss={record.get('val_loss'):.6f}"
        )
        self.query_one("#passage", Static).update(Panel(passage_text, title=title))


def main() -> None:
    parser = argparse.ArgumentParser(description="View colorized validation passages over time")
    parser.add_argument("--yaml_file", default="highlighted_passage.yaml")
    args = parser.parse_args()

    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    records = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or []
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"No records found in {yaml_path}")

    PassageViewer(records).run()


if __name__ == "__main__":
    main()
