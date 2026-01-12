"""Textual TUI to inspect Char-BPE vocabulary JSON exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Footer, Header, Static


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _token_length(token: str) -> int:
    if token.startswith("<byte:") and token.endswith(">"):
        return 1
    return len(token)


def load_vocab_rows(
    vocab_path: Optional[Path], counts_path: Optional[Path]
) -> List[tuple[str, int, int]]:
    """Load rows of (token, frequency, symbol_length) from JSON files."""

    tokens: List[str] = []
    if vocab_path and vocab_path.exists():
        vocab_data = _load_json(vocab_path)
        if not isinstance(vocab_data, list):
            raise ValueError(
                f"Expected a list in {vocab_path}, but received {type(vocab_data).__name__}."
            )
        for item in vocab_data:
            if not isinstance(item, str):
                raise ValueError(
                    f"Vocabulary entries must be strings, found {type(item).__name__}."
                )
            tokens.append(item)

    counts_map: dict[int, tuple[int, Optional[str]]] = {}
    if counts_path and counts_path.exists():
        counts_data = _load_json(counts_path)
        if not isinstance(counts_data, list):
            raise ValueError(
                f"Expected a list of objects in {counts_path}, got {type(counts_data).__name__}."
            )
        for entry in counts_data:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Each count entry must be an object, found {type(entry).__name__}."
                )
            idx = entry.get("id")
            if not isinstance(idx, int):
                raise ValueError("Token count entries must include integer 'id' fields.")
            count = entry.get("count", 0)
            if not isinstance(count, int):
                raise ValueError("Token count entries must include integer 'count' fields.")
            token_value = entry.get("token") if isinstance(entry.get("token"), str) else None
            counts_map[idx] = (count, token_value)

    rows: List[tuple[str, int, int]] = []
    if tokens:
        for idx, token in enumerate(tokens):
            count, override_token = counts_map.get(idx, (0, None))
            display_token = override_token or token
            rows.append((display_token, count, _token_length(display_token)))
    elif counts_map:
        for idx in sorted(counts_map):
            count, token_value = counts_map[idx]
            if token_value is None:
                token_value = f"<id:{idx}>"
            rows.append((token_value, count, _token_length(token_value)))
    else:
        raise FileNotFoundError(
            "No vocabulary or token-count JSON files were found. Provide at least one."
        )

    return rows


class VocabExplorer(App):
    """Simple Textual TUI for browsing Char-BPE vocabulary statistics."""

    CSS = """
    Screen { align: center middle; }
    Container { height: 1fr; width: 1fr; }
    DataTable { height: 1fr; }
    Static#empty { padding: 1 2; }
    """

    def __init__(self, rows: List[tuple[str, int, int]]) -> None:
        super().__init__()
        self._original_rows = rows
        self._display_rows = list(rows)
        self._sort_column: Optional[int] = None
        self._sort_ascending: bool = True
        self._table: Optional[DataTable] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        if not self._display_rows:
            yield Static("No vocabulary entries to display.", id="empty")
            yield Footer()
            return
        yield Container(DataTable(id="table", zebra_stripes=True))
        yield Footer()

    def on_mount(self) -> None:
        if not self._display_rows:
            return
        self._table = self.query_one(DataTable)
        self._table.add_columns("Token", "Frequency", "Length")
        self._table.cursor_type = "row"
        self._refresh_table()
        self._table.focus()

    def _refresh_table(self) -> None:
        if not self._table:
            return
        self._table.clear(columns=False)
        for token, count, length in self._display_rows:
            self._table.add_row(token, f"{count}", f"{length}")

    def _sort_rows(self, column_index: int) -> None:
        if column_index == self._sort_column:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = column_index
            self._sort_ascending = True

        key_funcs = (
            lambda row: row[0],
            lambda row: row[1],
            lambda row: row[2],
        )
        key_func = key_funcs[column_index]
        self._display_rows = sorted(
            self._display_rows,
            key=key_func,
            reverse=not self._sort_ascending,
        )
        self._refresh_table()

    def _clear_sort(self) -> None:
        self._display_rows = list(self._original_rows)
        self._sort_column = None
        self._sort_ascending = True
        self._refresh_table()

    def on_key(self, event: events.Key) -> None:
        if not self._table:
            return
        if event.key == "enter":
            column = self._table.cursor_column
            if column is not None:
                self._sort_rows(column)
                event.stop()
        elif event.key == "U" or (event.key == "u" and event.shift) or getattr(event, "character", "") == "U":
            self._clear_sort()
            event.stop()
        elif event.key == "q":
            self.exit()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Explore Char-BPE vocabulary JSON outputs.")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("char_bpe_vocab.json"),
        help="Path to the vocabulary JSON file (default: char_bpe_vocab.json).",
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("char_bpe_token_counts.json"),
        help="Path to the token counts JSON file (default: char_bpe_token_counts.json).",
    )
    args = parser.parse_args(argv)

    vocab_path = args.vocab if args.vocab and args.vocab.exists() else None
    counts_path = args.counts if args.counts and args.counts.exists() else None

    rows = load_vocab_rows(vocab_path, counts_path)
    app = VocabExplorer(rows)
    app.run()


if __name__ == "__main__":
    main()
