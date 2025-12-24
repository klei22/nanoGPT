import importlib.util
from typing import Iterable, Optional


class ZeusEnergyProfiler:
    def __init__(
        self,
        enabled: bool,
        target: str,
        gpu_indices: Optional[Iterable[int]] = None,
        cpu_indices: Optional[Iterable[int]] = None,
        sync_execution_with: Optional[str] = "torch",
    ) -> None:
        self.enabled = enabled
        self.target = target
        self._monitor = None

        if not enabled:
            return

        if importlib.util.find_spec("zeus") is None:
            print("Zeus is not installed; energy profiling is disabled.")
            self.enabled = False
            return

        from zeus.monitor import ZeusMonitor

        resolved_gpu_indices = list(gpu_indices) if gpu_indices is not None else None
        resolved_cpu_indices = list(cpu_indices) if cpu_indices is not None else None

        if target == "gpu":
            resolved_cpu_indices = []
        elif target == "cpu":
            resolved_gpu_indices = []

        try:
            self._monitor = ZeusMonitor(
                gpu_indices=resolved_gpu_indices,
                cpu_indices=resolved_cpu_indices,
                sync_execution_with=sync_execution_with,
            )
        except Exception as exc:
            print(f"Zeus profiler failed to initialize: {exc}")
            self.enabled = False
            self._monitor = None

    def begin(self, window_name: str) -> None:
        if not self.enabled or self._monitor is None:
            return
        self._monitor.begin_window(window_name)

    def end(self, window_name: str):
        if not self.enabled or self._monitor is None:
            return None
        return self._monitor.end_window(window_name)

    @staticmethod
    def total_energy_joules(measurement) -> Optional[float]:
        if measurement is None:
            return None
        total_energy = getattr(measurement, "total_energy", None)
        if total_energy is not None:
            return float(total_energy)
        energy_map = getattr(measurement, "energy", None)
        if isinstance(energy_map, dict) and energy_map:
            return float(sum(energy_map.values()))
        return None
