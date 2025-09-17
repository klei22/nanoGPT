"""Utilities for configuring the training progress bar display.

This module centralizes the available progress bar presets and the
keyboard shortcuts that toggle them. It also exposes a helper class
that keeps ``train.py`` free from the preset bookkeeping logic while
still allowing the training loop to react to user hotkeys.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


ColumnFactory = Callable[[], List]


class NonBlockingInput:
    """Cross-platform helper for reading single characters without blocking."""

    def __init__(self) -> None:
        self._enabled = False
        self._is_windows = os.name == "nt"
        self._msvcrt = None  # type: ignore[assignment]
        self._termios = None  # type: ignore[assignment]
        self._tty = None  # type: ignore[assignment]
        self._fcntl = None  # type: ignore[assignment]
        self._select = None  # type: ignore[assignment]
        self._old_term_settings = None
        self._old_flags = None

    def enable(self) -> bool:
        """Enable non-blocking single-character reads."""
        if self._enabled:
            return True
        if self._is_windows:
            try:
                import msvcrt  # type: ignore
            except ImportError:
                return False
            self._msvcrt = msvcrt
            self._enabled = True
            return True
        if not sys.stdin.isatty():
            return False
        try:
            import termios  # type: ignore
            import tty  # type: ignore
            import fcntl  # type: ignore
            import select  # type: ignore
        except ImportError:
            return False
        try:
            self._termios = termios
            self._tty = tty
            self._fcntl = fcntl
            self._select = select
            self._old_term_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._old_flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
            fcntl.fcntl(sys.stdin, fcntl.F_SETFL, self._old_flags | os.O_NONBLOCK)
            self._enabled = True
        except Exception:
            self.disable()
            return False
        return True

    def disable(self) -> None:
        """Restore terminal state if non-blocking mode was active."""
        if not self._enabled:
            return
        if self._is_windows:
            self._enabled = False
            self._msvcrt = None
            return
        try:
            if self._termios is not None and self._old_term_settings is not None:
                self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_term_settings)
            if self._fcntl is not None and self._old_flags is not None:
                self._fcntl.fcntl(sys.stdin, self._fcntl.F_SETFL, self._old_flags)
        finally:
            self._enabled = False
            self._termios = None
            self._tty = None
            self._fcntl = None
            self._select = None
            self._old_term_settings = None
            self._old_flags = None

    def get_key(self) -> Optional[str]:
        """Return the next pressed key if available, otherwise ``None``."""
        if not self._enabled:
            return None
        if self._is_windows:
            assert self._msvcrt is not None
            if self._msvcrt.kbhit():
                return self._msvcrt.getwch()
            return None
        if self._select is None:
            return None
        try:
            ready, _, _ = self._select.select([sys.stdin], [], [], 0)
        except (OSError, ValueError):
            return None
        if not ready:
            return None
        try:
            return sys.stdin.read(1)
        except (OSError, IOError):
            return None


class ProgressDisplaySettings:
    """Container for the available progress bar presets and hotkeys."""

    def __init__(self) -> None:
        self.default_mode = "all"
        self.hotkey_to_mode: "OrderedDict[str, str]" = OrderedDict(
            [
                ("a", "all"),
                ("t", "time"),
                ("s", "stats"),
                ("m", "stats_min"),
            ]
        )
        self.cycle_hotkeys: List[str] = ["c"]
        self.cycle_order: List[str] = ["all", "time", "stats", "stats_min"]
        self._mode_factories: Dict[str, List[ColumnFactory]] = {
            "all": [
                self._metrics_columns,
                self._time_columns,
                self._gpu_columns,
                self._full_stats_columns,
            ],
            "time": [self._metrics_columns, self._time_columns],
            "stats": [self._metrics_columns, self._gpu_columns, self._full_stats_columns],
            "stats_min": [self._metrics_columns, self._gpu_columns, self._minimal_stats_columns],
        }

    @property
    def mode_names(self) -> Iterable[str]:
        return self._mode_factories.keys()

    def columns_for_mode(self, mode: str) -> List:
        target = mode if mode in self._mode_factories else self.default_mode
        columns = self._base_columns()
        for factory in self._mode_factories[target]:
            columns.extend(factory())
        return columns

    def next_mode(self, current_mode: str) -> str:
        if not self.cycle_order:
            return self.default_mode
        try:
            index = self.cycle_order.index(current_mode)
        except ValueError:
            return self.default_mode
        return self.cycle_order[(index + 1) % len(self.cycle_order)]

    def render_hotkey_help(self) -> str:
        direct_parts = [
            f"[{key}] {self._format_mode_name(mode)}" for key, mode in self.hotkey_to_mode.items()
        ]
        cycle_parts = [f"[{key}] cycle" for key in self.cycle_hotkeys]
        segments: List[str] = []
        if cycle_parts:
            segments.append("cycle: " + ", ".join(cycle_parts))
        if direct_parts:
            segments.append("direct: " + " | ".join(direct_parts))
        return "; ".join(segments)

    def _format_mode_name(self, mode: str) -> str:
        return mode.replace("_", " ")

    def _base_columns(self) -> List:
        return [
            TextColumn("[bold white]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Mode:[/bold yellow]{task.fields[display_mode]}")
        ]

    def _metrics_columns(self) -> List:
        return [
            TextColumn(
                "-- [bold dark_cyan]BestIter:[/bold dark_cyan]{task.fields[best_iter]} "
                "[bold dark_cyan]BestValLoss:[/bold dark_cyan]{task.fields[best_val_loss]}"
            )
        ]

    def _time_columns(self) -> List:
        return [
            TimeRemainingColumn(compact=False),
            TextColumn("-- [bold purple3]ETA:[/bold purple3]{task.fields[eta]}"),
            TextColumn("[bold purple3]Remaining:[/bold purple3]{task.fields[hour]}h{task.fields[min]}m"),
            TextColumn("[bold purple3]total_est:[/bold purple3]{task.fields[total_hour]}h{task.fields[total_min]}m"),
            TextColumn("-- [bold dark_magenta]iter_latency:[/bold dark_magenta]{task.fields[iter_latency]}ms"),
        ]

    def _gpu_columns(self) -> List:
        return [TextColumn("[bold dark_magenta]peak_gpu_mb:[/bold dark_magenta]{task.fields[peak_gpu_mb]}MB")]

    def _full_stats_columns(self) -> List:
        return [
            TextColumn("-- [bold dark_cyan]T1P:[/bold dark_cyan]{task.fields[t1p]}"),
            TextColumn("[bold dark_cyan]T1C:[/bold dark_cyan]{task.fields[t1c]}"),
            TextColumn("-- [bold dark_magenta]TR:[/bold dark_magenta]{task.fields[tr]}"),
            TextColumn("[bold dark_magenta]TP:[/bold dark_magenta]{task.fields[tp]}"),
            TextColumn("[bold dark_magenta]TLP:[/bold dark_magenta]{task.fields[tlp]}"),
            TextColumn("[bold dark_magenta]R95:[/bold dark_magenta]{task.fields[r95]}"),
            TextColumn("[bold dark_magenta]P95:[/bold dark_magenta]{task.fields[p95]}"),
        ]

    def _minimal_stats_columns(self) -> List:
        return [
            TextColumn("-- [bold dark_cyan]T1P:[/bold dark_cyan]{task.fields[t1p]}"),
            TextColumn("[bold dark_cyan]T1C:[/bold dark_cyan]{task.fields[t1c]}"),
            TextColumn("-- [bold dark_magenta]TR:[/bold dark_magenta]{task.fields[tr]}"),
        ]


class ProgressDisplayManager:
    """Manage the active progress bar mode and handle user hotkeys."""

    def __init__(self, console) -> None:
        self.console = console
        self.settings = ProgressDisplaySettings()
        self._current_mode = self.settings.default_mode
        self.progress = Progress(*self.settings.columns_for_mode(self._current_mode), console=self.console)
        self._input = NonBlockingInput()
        self._listening = False

    def start(self) -> bool:
        """Begin listening for hotkeys. Returns ``True`` if enabled."""
        self._listening = self._input.enable()
        return self._listening

    def stop(self) -> None:
        """Stop listening for hotkeys and restore terminal state."""
        self._input.disable()
        self._listening = False

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def help_message(self) -> str:
        if self._listening:
            return f"Progress hotkeys: {self.settings.render_hotkey_help()}"
        return "Progress hotkeys unavailable (requires an interactive terminal)."

    def poll(self) -> Optional[str]:
        """Check for user input and update the progress display if needed."""
        if not self._listening:
            return None
        key = self._input.get_key()
        if not key:
            return None
        return self._handle_key(key)

    def _handle_key(self, key: str) -> Optional[str]:
        normalized = key.lower()
        target_mode = None
        if normalized in self.settings.hotkey_to_mode:
            target_mode = self.settings.hotkey_to_mode[normalized]
        elif normalized in self.settings.cycle_hotkeys:
            target_mode = self.settings.next_mode(self._current_mode)
        if not target_mode:
            return None
        return self._set_mode(target_mode)

    def _set_mode(self, mode: str) -> Optional[str]:
        if mode not in self.settings.mode_names:
            return None
        if mode == self._current_mode:
            return None
        self._current_mode = mode
        new_columns = self.settings.columns_for_mode(mode)
        try:
            self.progress.columns = new_columns
        except AttributeError:
            setattr(self.progress, "columns", new_columns)
        self.progress.refresh()
        return mode


__all__ = ["ProgressDisplayManager", "ProgressDisplaySettings"]
