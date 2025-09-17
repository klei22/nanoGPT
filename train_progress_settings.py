"""Progress bar presets and hotkey handling for train.py."""
from __future__ import annotations

import atexit
import select
import sys
from dataclasses import dataclass
from typing import Callable, List, Sequence

try:  # pragma: no cover - platform-dependent behavior
    import termios
    import tty
except ImportError:  # pragma: no cover - platform-dependent behavior
    termios = None  # type: ignore
    tty = None  # type: ignore

from rich.progress import (
    BarColumn,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

ColumnFactory = Callable[[], ProgressColumn]


def _description_column() -> ProgressColumn:
    return TextColumn("[bold white]{task.description}")


def _bar_column() -> ProgressColumn:
    return BarColumn()


def _task_progress_column() -> ProgressColumn:
    return TaskProgressColumn()


def _time_remaining_column() -> ProgressColumn:
    return TimeRemainingColumn(compact=False)


def _best_iter_column() -> ProgressColumn:
    return TextColumn(
        "-- [bold dark_cyan]BestIter:[/bold dark_cyan]{task.fields[best_iter]} "
        "[bold dark_cyan]BestValLoss:[/bold dark_cyan]{task.fields[best_val_loss]}"
    )


def _eta_column() -> ProgressColumn:
    return TextColumn("-- [bold purple3]ETA:[/bold purple3]{task.fields[eta]}")


def _remaining_column() -> ProgressColumn:
    return TextColumn(
        "[bold purple3]Remaining:[/bold purple3]{task.fields[hour]}h{task.fields[min]}m"
    )


def _total_est_column() -> ProgressColumn:
    return TextColumn(
        "[bold purple3]total_est:[/bold purple3]{task.fields[total_hour]}h{task.fields[total_min]}m"
    )


def _iter_latency_column() -> ProgressColumn:
    return TextColumn(
        "-- [bold dark_magenta]iter_latency:[/bold dark_magenta]{task.fields[iter_latency]}ms"
    )


def _peak_gpu_column() -> ProgressColumn:
    return TextColumn(
        "[bold dark_magenta]peak_gpu_mb:[/bold dark_magenta]{task.fields[peak_gpu_mb]}MB"
    )


def _top1_prob_column() -> ProgressColumn:
    return TextColumn("-- [bold dark_cyan]T1P:[/bold dark_cyan]{task.fields[t1p]}")


def _top1_correct_column() -> ProgressColumn:
    return TextColumn("[bold dark_cyan]T1C:[/bold dark_cyan]{task.fields[t1c]}")


def _target_rank_column() -> ProgressColumn:
    return TextColumn("-- [bold dark_magenta]TR:[/bold dark_magenta]{task.fields[tr]}")


def _target_prob_column() -> ProgressColumn:
    return TextColumn("[bold dark_magenta]TP:[/bold dark_magenta]{task.fields[tp]}")


def _target_left_prob_column() -> ProgressColumn:
    return TextColumn("[bold dark_magenta]TLP:[/bold dark_magenta]{task.fields[tlp]}")


def _rank_95_column() -> ProgressColumn:
    return TextColumn("[bold dark_magenta]R95:[/bold dark_magenta]{task.fields[r95]}")


def _left_prob_95_column() -> ProgressColumn:
    return TextColumn("[bold dark_magenta]P95:[/bold dark_magenta]{task.fields[p95]}")


ALL_COLUMN_FACTORIES: Sequence[ColumnFactory] = (
    _time_remaining_column,
    _best_iter_column,
    _eta_column,
    _remaining_column,
    _total_est_column,
    _iter_latency_column,
    _peak_gpu_column,
    _top1_prob_column,
    _top1_correct_column,
    _target_rank_column,
    _target_prob_column,
    _target_left_prob_column,
    _rank_95_column,
    _left_prob_95_column,
)

TIME_COLUMN_FACTORIES: Sequence[ColumnFactory] = (
    _time_remaining_column,
    _eta_column,
    _remaining_column,
    _total_est_column,
)

STATS_DETAILED_FACTORIES: Sequence[ColumnFactory] = (
    _best_iter_column,
    _iter_latency_column,
    _peak_gpu_column,
    _top1_prob_column,
    _top1_correct_column,
    _target_rank_column,
    _target_prob_column,
    _target_left_prob_column,
    _rank_95_column,
    _left_prob_95_column,
)

STATS_MINIMAL_FACTORIES: Sequence[ColumnFactory] = (
    _best_iter_column,
    _top1_prob_column,
    _target_rank_column,
)

MINIMAL_FACTORIES: Sequence[ColumnFactory] = (
    _time_remaining_column,
)

BASE_FACTORIES: Sequence[ColumnFactory] = (
    _description_column,
    _bar_column,
    _task_progress_column,
)

DEFAULT_PRESET_KEY = "a"

NEXT_CYCLE_KEYS = ("c", "n")
PREVIOUS_CYCLE_KEYS = ("p", "b")


@dataclass(frozen=True)
class ProgressPreset:
    """Definition for a progress display preset."""

    key: str
    name: str
    description: str
    column_factories: Sequence[ColumnFactory]


PRESETS: Sequence[ProgressPreset] = (
    ProgressPreset(
        key="a",
        name="All metrics",
        description="Display all timing, iteration, and evaluation statistics.",
        column_factories=ALL_COLUMN_FACTORIES,
    ),
    ProgressPreset(
        key="t",
        name="Timing focus",
        description="Show only timing-related estimates.",
        column_factories=TIME_COLUMN_FACTORIES,
    ),
    ProgressPreset(
        key="s",
        name="Stats focus",
        description="Emphasize validation statistics and iteration diagnostics.",
        column_factories=STATS_DETAILED_FACTORIES,
    ),
    ProgressPreset(
        key="r",
        name="Stats minimal",
        description="Show a compact selection of the most important stats.",
        column_factories=STATS_MINIMAL_FACTORIES,
    ),
    ProgressPreset(
        key="m",
        name="Minimal",
        description="Display only the progress bar and time remaining.",
        column_factories=MINIMAL_FACTORIES,
    ),
)


class ProgressDisplayManager:
    """Manage progress bar presets and keyboard input for train.py."""

    def __init__(self, console, enable_hotkeys: bool = True):
        self.console = console
        self.presets: Sequence[ProgressPreset] = PRESETS
        self._hotkey_to_index = {
            preset.key.lower(): index for index, preset in enumerate(self.presets)
        }
        self._cycle_key_map = {
            **{key: 1 for key in NEXT_CYCLE_KEYS},
            **{key: -1 for key in PREVIOUS_CYCLE_KEYS},
        }
        default_index = self._hotkey_to_index.get(DEFAULT_PRESET_KEY, 0)
        self._current_index = default_index
        self._terminal_configured = False
        self._terminal_fd = None
        self._old_terminal_settings = None

        self._hotkeys_enabled = enable_hotkeys and sys.stdin.isatty()
        if self._hotkeys_enabled and termios is not None and tty is not None:
            try:
                self._terminal_fd = sys.stdin.fileno()
                self._old_terminal_settings = termios.tcgetattr(self._terminal_fd)
                tty.setcbreak(self._terminal_fd)
                self._terminal_configured = True
                atexit.register(self.close)
            except Exception:
                # Fallback gracefully if the terminal can't be configured
                self._hotkeys_enabled = False
                self._terminal_configured = False
        else:
            self._hotkeys_enabled = False

    @property
    def current_preset(self) -> ProgressPreset:
        return self.presets[self._current_index]

    def build_current_columns(self) -> List[ProgressColumn]:
        columns: List[ProgressColumn] = [factory() for factory in BASE_FACTORIES]
        columns.extend(factory() for factory in self.current_preset.column_factories)
        return columns

    def _read_key(self) -> str | None:
        if not self._hotkeys_enabled:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            self._hotkeys_enabled = False
            self.close()
            return None
        if not ready:
            return None
        try:
            key = sys.stdin.read(1)
        except Exception:
            self._hotkeys_enabled = False
            self.close()
            return None
        return key

    def _handle_key(self, key: str) -> bool:
        if not key:
            return False
        lower_key = key.lower()
        if lower_key in self._hotkey_to_index:
            new_index = self._hotkey_to_index[lower_key]
            if new_index != self._current_index:
                self._current_index = new_index
                return True
            return False
        if lower_key in self._cycle_key_map:
            direction = self._cycle_key_map[lower_key]
            self._current_index = (self._current_index + direction) % len(self.presets)
            return True
        return False

    def poll(self) -> bool:
        """Poll for new keyboard input; returns True if preset changed."""

        if not self._hotkeys_enabled:
            return False

        changed = False
        while True:
            key = self._read_key()
            if key is None:
                break
            changed = self._handle_key(key) or changed
        return changed

    def build_controls_text(self) -> Text:
        """Return a Text renderable describing the available presets."""

        if not self._hotkeys_enabled:
            return Text.from_markup(
                "[dim]Progress display hotkeys unavailable (stdin is not a TTY or could not be configured).[/dim]"
            )

        next_keys = ", ".join(f"'{key}'" for key in NEXT_CYCLE_KEYS)
        prev_keys = ", ".join(f"'{key}'" for key in PREVIOUS_CYCLE_KEYS)
        direct_keys = ", ".join(
            f"[bold]{preset.key}[/bold]: {preset.name}" for preset in self.presets
        )
        lines = [
            "[bold bright_cyan]Progress display preset:[/bold bright_cyan] "
            f"{self.current_preset.name}",
            f"[dim]{self.current_preset.description}[/dim]",
            f"Cycle presets: {next_keys} for next, {prev_keys} for previous",
            f"Direct presets: {direct_keys}",
        ]
        return Text.from_markup("\n".join(lines))

    def close(self) -> None:
        """Restore the terminal configuration if it was modified."""

        if self._terminal_configured and self._terminal_fd is not None:
            try:
                termios.tcsetattr(
                    self._terminal_fd,
                    termios.TCSADRAIN,
                    self._old_terminal_settings,  # type: ignore[arg-type]
                )
            except Exception:
                pass
            finally:
                self._terminal_configured = False

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()


__all__ = ["ProgressDisplayManager", "ProgressPreset", "PRESETS"]
