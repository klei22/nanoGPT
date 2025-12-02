import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnergyMetrics:
    overall_joules: float = float("nan")
    attention_joules: float = float("nan")
    mlp_joules: float = float("nan")
    lm_head_joules: float = float("nan")
    tokens: int = 0

    def per_token(self) -> Dict[str, float]:
        if self.tokens <= 0:
            return {
                "overall": float("nan"),
                "attention": float("nan"),
                "mlp": float("nan"),
                "lm_head": float("nan"),
            }
        return {
            "overall": self.overall_joules / self.tokens,
            "attention": self.attention_joules / self.tokens,
            "mlp": self.mlp_joules / self.tokens,
            "lm_head": self.lm_head_joules / self.tokens,
        }


class EnergyTracker:
    """Utility that wraps Zeus to track GPU energy usage per training iteration.

    The tracker is intentionally lightweight and becomes a no-op when Zeus isn't
    installed or when energy tracking is disabled via CLI flags.
    """

    _CATEGORIES: Iterable[str] = ("attention", "mlp", "lm_head")

    def __init__(self, enabled: bool, device_index: int = 0) -> None:
        self.enabled = enabled
        self.device_index = device_index
        self.monitor = None
        self.power_monitor = None
        self.active_window: Optional[str] = None
        self.window_counter = 0
        self.current_tokens: int = 0
        self.current_totals: defaultdict[str, float] = defaultdict(float)
        self.accumulated_totals: defaultdict[str, float] = defaultdict(float)
        self.accumulated_tokens: int = 0
        self._latest_metrics = EnergyMetrics()
        self._global_average: Dict[str, float] = {
            "overall": float("nan"),
            "attention": float("nan"),
            "mlp": float("nan"),
            "lm_head": float("nan"),
        }
        self._open_windows: dict[object, str] = {}

        if not self.enabled:
            return

        try:  # Lazy import to avoid hard dependency when flag is off
            from zeus.monitor import ZeusMonitor, PowerMonitor  # type: ignore

            self.power_monitor = PowerMonitor(device_indices=[self.device_index])
            self.monitor = ZeusMonitor(device_indices=[self.device_index])
            self.enabled = True
            logger.info("Zeus energy tracking enabled on device %s", self.device_index)
        except Exception as exc:  # pragma: no cover - best-effort import guard
            logger.warning("Zeus unavailable, disabling energy tracking: %s", exc)
            self.enabled = False

    # ------------------------------------------------------------------
    # Hook registration
    def attach_model(self, model) -> None:
        if not self.enabled:
            return

        if "h" in model.transformer:
            for block in model.transformer["h"]:
                if hasattr(block, "attn"):
                    block.attn.register_forward_pre_hook(self._make_pre_hook("attention"))
                    block.attn.register_forward_hook(self._make_post_hook("attention"))
                if hasattr(block, "mlp"):
                    block.mlp.register_forward_pre_hook(self._make_pre_hook("mlp"))
                    block.mlp.register_forward_hook(self._make_post_hook("mlp"))

        # LM heads can be a single module or a ModuleDict keyed by dataset
        lm_heads = []
        if hasattr(model, "lm_head"):
            lm_heads.append(model.lm_head)
        if hasattr(model, "transformer"):
            for name, module in model.transformer.items():
                if name.startswith("lm_head"):
                    lm_heads.append(module)

        for head in lm_heads:
            head.register_forward_pre_hook(self._make_pre_hook("lm_head"))
            head.register_forward_hook(self._make_post_hook("lm_head"))

    # ------------------------------------------------------------------
    # Iteration lifecycle helpers
    def start_iteration(self) -> None:
        if not self.enabled:
            return
        self.active_window = f"train_iteration_{self.window_counter}"
        self.window_counter += 1
        self.current_tokens = 0
        self.current_totals.clear()
        self.monitor.begin_window(self.active_window)

    def add_tokens(self, tokens: int) -> None:
        if not self.enabled:
            return
        self.current_tokens += int(tokens)

    def end_iteration(self) -> Optional[EnergyMetrics]:
        if not self.enabled or not self.active_window:
            return None

        measurement = self.monitor.end_window(self.active_window)
        total_energy = getattr(measurement, "total_energy", float("nan"))

        metrics = EnergyMetrics(
            overall_joules=total_energy,
            attention_joules=self.current_totals.get("attention", float("nan")),
            mlp_joules=self.current_totals.get("mlp", float("nan")),
            lm_head_joules=self.current_totals.get("lm_head", float("nan")),
            tokens=self.current_tokens,
        )
        self._latest_metrics = metrics
        self.active_window = None

        self._update_aggregates(metrics)
        return metrics

    def latest_iteration_per_token(self) -> Dict[str, float]:
        return self._latest_metrics.per_token()

    def global_average_per_token(self) -> Dict[str, float]:
        return self._global_average

    # ------------------------------------------------------------------
    # Internal helpers
    def _update_aggregates(self, metrics: EnergyMetrics) -> None:
        if metrics.tokens > 0:
            self.accumulated_tokens += metrics.tokens
        for key, value in zip(
            ("overall", *self._CATEGORIES),
            (
                metrics.overall_joules,
                metrics.attention_joules,
                metrics.mlp_joules,
                metrics.lm_head_joules,
            ),
        ):
            if value is not None and not math.isnan(value):
                self.accumulated_totals[key] += value

        if self.accumulated_tokens > 0:
            self._global_average = {
                k: self.accumulated_totals.get(k, float("nan")) / self.accumulated_tokens
                for k in ("overall", *self._CATEGORIES)
            }
        else:
            self._global_average = {k: float("nan") for k in ("overall", *self._CATEGORIES)}

    def _make_pre_hook(self, category: str):
        def _pre_hook(module, _input):  # pragma: no cover - tiny wrapper
            if not self.enabled or not self.active_window:
                return
            window_name = f"{category}_{id(module)}_{self.window_counter}"
            self._open_windows[module] = window_name
            self.monitor.begin_window(window_name)

        return _pre_hook

    def _make_post_hook(self, category: str):
        def _post_hook(module, _input, _output):  # pragma: no cover - tiny wrapper
            if not self.enabled or not self.active_window:
                return
            window_name = self._open_windows.pop(module, None)
            if window_name is None:
                return
            measurement = self.monitor.end_window(window_name)
            if measurement is None:
                return
            self.current_totals[category] += getattr(measurement, "total_energy", 0.0)

        return _post_hook
