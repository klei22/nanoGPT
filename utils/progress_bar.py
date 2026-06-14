"""Helpers for building the training progress bar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn


@dataclass(frozen=True)
class ProgressMetric:
    """Declarative configuration for one metric displayed in the progress bar."""

    field: str
    label: str
    style: str
    formatter: Callable[[object], str]
    suffix: str = ""
    prefix: str = ""

    def column(self) -> TextColumn:
        return TextColumn(
            f"{self.prefix}[bold {self.style}]{self.label}:[/bold {self.style}]"
            f"{{task.fields[{self.field}]}}{self.suffix}"
        )


DEFAULT_PROGRESS_METRICS: list[ProgressMetric] = [
    ProgressMetric("best_iter", "BestIter", "dark_cyan", lambda trainer: f"{trainer.best_iter}", prefix="-- "),
    ProgressMetric("best_val_loss", "BestValLoss", "dark_cyan", lambda trainer: f"{trainer.best_val_loss:.3f}"),
    ProgressMetric("best_tokens", "BestTokens", "dark_cyan", lambda trainer: f"{trainer.best_tokens}"),
    ProgressMetric("eta", "ETA", "purple3", lambda trainer: trainer.formatted_completion_eta, prefix="-- "),
    ProgressMetric(
        "remaining",
        "Remaining",
        "purple3",
        lambda trainer: f"{int((trainer.time_remaining_ms // 3_600_000) % 24):02d}h"
        f"{int((trainer.time_remaining_ms // 60_000) % 60):02d}m",
    ),
    ProgressMetric(
        "total_est",
        "total_est",
        "purple3",
        lambda trainer: f"{int(trainer.total_time_est_ms // 3_600_000)}h"
        f"{int((trainer.total_time_est_ms // 60_000) % 60):02d}m",
    ),
    ProgressMetric("iter_latency", "iter_latency", "dark_magenta", lambda trainer: f"{trainer.iter_latency_avg:.1f}", suffix="ms", prefix="-- "),
    ProgressMetric(
        "peak_gpu_mb",
        "peak_gpu_mb",
        "dark_magenta",
        lambda trainer: f"{trainer.peak_torch_allocated / (1024 ** 2):.1f}",
        suffix="MB",
    ),
    ProgressMetric("top1_prob", "T1P", "dark_cyan", lambda trainer: f"{trainer.latest_top1_prob:.6f}", prefix="-- "),
    ProgressMetric("target_rank", "TR", "dark_magenta", lambda trainer: f"{trainer.latest_target_rank:.2f}", prefix="-- "),
]


def build_training_progress(console: Console) -> Progress:
    """Build the default Rich progress bar used during training."""

    return Progress(
        TextColumn("[bold white]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(compact=False),
        *(metric.column() for metric in DEFAULT_PROGRESS_METRICS),
        console=console,
    )


def build_progress_fields(trainer: object) -> dict[str, str]:
    """Format all default progress metric fields for a trainer instance."""

    return {metric.field: metric.formatter(trainer) for metric in DEFAULT_PROGRESS_METRICS}
