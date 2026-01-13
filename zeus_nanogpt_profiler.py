from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import time
from typing import Optional, Dict, Any

import torch


# ---- Robust imports across Zeus versions / layouts ----
def _import_zeus_monitors():
    # ZeusMonitor
    try:
        from zeus.monitor import ZeusMonitor  # type: ignore
    except Exception:
        from zeus.monitor.energy import ZeusMonitor  # type: ignore

    # PowerMonitor
    try:
        from zeus.monitor import PowerMonitor  # type: ignore
    except Exception:
        from zeus.monitor.power import PowerMonitor  # type: ignore

    # TemperatureMonitor
    try:
        from zeus.monitor import TemperatureMonitor  # type: ignore
    except Exception:
        from zeus.monitor.temperature import TemperatureMonitor  # type: ignore

    return ZeusMonitor, PowerMonitor, TemperatureMonitor


@dataclass
class ZeusNanoGPTConfig:
    enabled: bool = False
    out_dir: str = "out"
    window_prefix: str = "nanogpt"

    # which CUDA device index *from the framework perspective* (after CUDA_VISIBLE_DEVICES)
    gpu_index: Optional[int] = None

    # profiling cadence
    profile_every: int = 1
    warmup_iters: int = 0

    # timelines
    power_trace: bool = True
    power_update_period: float = 0.2
    power_max_samples_per_gpu: Optional[int] = 200000
    power_domains: Optional[list[str]] = None  # let Zeus auto-detect if None

    temperature_trace: bool = True
    temperature_update_period: float = 1.0
    temperature_max_samples_per_gpu: Optional[int] = 20000


class ZeusNanoGPTProfiler:
    """
    nanoGPT-oriented wrapper around ZeusMonitor + PowerMonitor + TemperatureMonitor.
    Writes:
      - zeus_windows.csv (per-window energy/time)
      - zeus_power.csv (power timelines)
      - zeus_temperature.csv (temperature timelines)
      - zeus_summary.json (total training window)
    """

    def __init__(self, cfg: ZeusNanoGPTConfig):
        self.cfg = cfg
        self._started = False
        self._finished = False

        self.ZeusMonitor = None
        self.PowerMonitor = None
        self.TemperatureMonitor = None

        self.energy_mon = None
        self.power_mon = None
        self.temp_mon = None

        self.run_t0_wall: Optional[float] = None

        # current-iter bookkeeping
        self._iter_active = False
        self._iter_num: Optional[int] = None
        self._iter_tokens: Optional[int] = None
        self._iter_t0_wall: Optional[float] = None
        self._iter_t1_wall: Optional[float] = None

        # outputs
        self.prof_dir: Optional[Path] = None
        self.windows_csv_f = None
        self.windows_writer: Optional[csv.DictWriter] = None

    def start(self):
        if not self.cfg.enabled or self._started:
            return

        try:
            self.ZeusMonitor, self.PowerMonitor, self.TemperatureMonitor = _import_zeus_monitors()
        except Exception:
            # Zeus not installed; disable profiling gracefully
            self.cfg.enabled = False
            return

        gpu = self.cfg.gpu_index
        if gpu is None:
            gpu = torch.cuda.current_device()

        out_base = Path(self.cfg.out_dir)
        self.prof_dir = out_base / "zeus"
        self.prof_dir.mkdir(parents=True, exist_ok=True)

        # per-window CSV
        windows_path = self.prof_dir / "zeus_windows.csv"
        self.windows_csv_f = windows_path.open("w", newline="")
        fields = [
            "iter",
            "stage",
            "t_start_s",
            "t_end_s",
            "time_s",
            "energy_j",
            "gpu_energy_j",
            "avg_power_w",
            "tokens",
            "j_per_token",
        ]
        self.windows_writer = csv.DictWriter(self.windows_csv_f, fieldnames=fields)
        self.windows_writer.writeheader()

        # Start monitors inside main() (NOT at import time)
        self.energy_mon = self.ZeusMonitor(gpu_indices=[gpu])
        self.run_t0_wall = time.time()

        # whole-run window
        self.energy_mon.begin_window(f"{self.cfg.window_prefix}.train_total")

        # continuous traces
        if self.cfg.power_trace:
            kwargs = dict(
                gpu_indices=[gpu],
                update_period=self.cfg.power_update_period,
                max_samples_per_gpu=self.cfg.power_max_samples_per_gpu,
            )
            if self.cfg.power_domains is not None:
                kwargs["power_domains"] = self.cfg.power_domains
            self.power_mon = self.PowerMonitor(**kwargs)

        if self.cfg.temperature_trace:
            self.temp_mon = self.TemperatureMonitor(
                gpu_indices=[gpu],
                update_period=self.cfg.temperature_update_period,
                max_samples_per_gpu=self.cfg.temperature_max_samples_per_gpu,
            )

        self._started = True

    def _rel(self, wall_t: float) -> float:
        assert self.run_t0_wall is not None
        return wall_t - self.run_t0_wall

    def _write_window(
        self,
        iter_num: Optional[int],
        stage: str,
        t0_wall: float,
        t1_wall: float,
        zeus_time_s: float,
        zeus_energy_total_j: float,
        zeus_gpu_energy_j: float,
        tokens: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if self.windows_writer is None:
            return None

        avg_power = zeus_energy_total_j / zeus_time_s if zeus_time_s > 0 else None
        j_per_token = (zeus_energy_total_j / tokens) if (tokens is not None and tokens > 0) else None

        row = dict(
            iter=iter_num,
            stage=stage,
            t_start_s=self._rel(t0_wall),
            t_end_s=self._rel(t1_wall),
            time_s=zeus_time_s,
            energy_j=zeus_energy_total_j,
            gpu_energy_j=zeus_gpu_energy_j,
            avg_power_w=avg_power,
            tokens=tokens,
            j_per_token=j_per_token,
        )
        self.windows_writer.writerow(row)
        self.windows_csv_f.flush()
        return row

    def begin_iter(self, iter_num: int, tokens_this_iter: int):
        if not self.cfg.enabled or not self._started:
            return

        if iter_num < self.cfg.warmup_iters:
            self._iter_active = False
            return

        if self.cfg.profile_every > 1 and (iter_num % self.cfg.profile_every != 0):
            self._iter_active = False
            return

        self._iter_active = True
        self._iter_num = iter_num
        self._iter_tokens = tokens_this_iter
        self._iter_t0_wall = time.time()

        # Outer iter window
        self.energy_mon.begin_window(f"{self.cfg.window_prefix}.iter_total")
        # Nested fwd/bwd window
        self.energy_mon.begin_window(f"{self.cfg.window_prefix}.fwdbwd")

    def end_fwdbwd(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled or not self._iter_active:
            return None

        m = self.energy_mon.end_window(f"{self.cfg.window_prefix}.fwdbwd")
        t1 = time.time()

        # single-GPU convenience
        gpu_energy = next(iter(m.gpu_energy.values())) if m.gpu_energy else 0.0

        row = self._write_window(
            iter_num=self._iter_num,
            stage="fwdbwd",
            t0_wall=self._iter_t0_wall,
            t1_wall=t1,
            zeus_time_s=m.time,
            zeus_energy_total_j=m.total_energy,
            zeus_gpu_energy_j=gpu_energy,
            tokens=self._iter_tokens,
        )

        # start nested opt step window
        self.energy_mon.begin_window(f"{self.cfg.window_prefix}.opt_step")
        return row

    def end_opt_step(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled or not self._iter_active:
            return None

        m = self.energy_mon.end_window(f"{self.cfg.window_prefix}.opt_step")
        t1 = time.time()
        gpu_energy = next(iter(m.gpu_energy.values())) if m.gpu_energy else 0.0

        return self._write_window(
            iter_num=self._iter_num,
            stage="opt_step",
            t0_wall=self._iter_t0_wall,
            t1_wall=t1,
            zeus_time_s=m.time,
            zeus_energy_total_j=m.total_energy,
            zeus_gpu_energy_j=gpu_energy,
            tokens=self._iter_tokens,
        )

    def end_iter(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled or not self._iter_active:
            return None

        m = self.energy_mon.end_window(f"{self.cfg.window_prefix}.iter_total")
        self._iter_t1_wall = time.time()
        gpu_energy = next(iter(m.gpu_energy.values())) if m.gpu_energy else 0.0

        row = self._write_window(
            iter_num=self._iter_num,
            stage="iter_total",
            t0_wall=self._iter_t0_wall,
            t1_wall=self._iter_t1_wall,
            zeus_time_s=m.time,
            zeus_energy_total_j=m.total_energy,
            zeus_gpu_energy_j=gpu_energy,
            tokens=self._iter_tokens,
        )

        self._iter_active = False
        return row

    def profile_block(self, name: str, fn, iter_num: Optional[int] = None):
        """
        Measure an arbitrary block (e.g. eval) as its own window.
        NOTE: overlaps with train_total window, which is fine because the names differ.
        """
        if not self.cfg.enabled or not self._started:
            return fn()

        t0 = time.time()
        self.energy_mon.begin_window(f"{self.cfg.window_prefix}.{name}")
        out = fn()
        m = self.energy_mon.end_window(f"{self.cfg.window_prefix}.{name}")
        t1 = time.time()

        gpu_energy = next(iter(m.gpu_energy.values())) if m.gpu_energy else 0.0

        self._write_window(
            iter_num=iter_num,
            stage=name,
            t0_wall=t0,
            t1_wall=t1,
            zeus_time_s=m.time,
            zeus_energy_total_j=m.total_energy,
            zeus_gpu_energy_j=gpu_energy,
            tokens=None,
        )
        return out

    def finish(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled or not self._started or self._finished:
            return None

        t_end = time.time()

        # end whole-run window
        train_total = self.energy_mon.end_window(f"{self.cfg.window_prefix}.train_total")
        gpu_energy = next(iter(train_total.gpu_energy.values())) if train_total.gpu_energy else 0.0

        summary = dict(
            time_s=train_total.time,
            total_energy_j=train_total.total_energy,
            gpu_energy_j=gpu_energy,
            gpu_energy_by_index=train_total.gpu_energy,
            wall_duration_s=(t_end - (self.run_t0_wall or t_end)),
        )
        (self.prof_dir / "zeus_summary.json").write_text(json.dumps(summary, indent=2))

        # export power timeline
        if self.power_mon is not None:
            timelines = self.power_mon.get_all_power_timelines(
                start_time=self.run_t0_wall,
                end_time=t_end,
            )
            power_path = self.prof_dir / "zeus_power.csv"
            with power_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["domain", "gpu_index", "t_s", "power_w"])
                w.writeheader()
                for domain, per_gpu in timelines.items():
                    for gpu_idx, samples in per_gpu.items():
                        for ts, pw in samples:
                            w.writerow(
                                dict(domain=domain, gpu_index=gpu_idx, t_s=(ts - self.run_t0_wall), power_w=pw)
                            )
            # stop background processes
            self.power_mon.stop()

        # export temperature timeline
        if self.temp_mon is not None:
            temps = self.temp_mon.get_temperature_timeline(
                start_time=self.run_t0_wall,
                end_time=t_end,
            )
            temp_path = self.prof_dir / "zeus_temperature.csv"
            with temp_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["gpu_index", "t_s", "temp_c"])
                w.writeheader()
                for gpu_idx, samples in temps.items():
                    for ts, temp_c in samples:
                        w.writerow(dict(gpu_index=gpu_idx, t_s=(ts - self.run_t0_wall), temp_c=temp_c))
            self.temp_mon.stop()

        if self.windows_csv_f is not None:
            self.windows_csv_f.close()

        self._finished = True
        return summary
