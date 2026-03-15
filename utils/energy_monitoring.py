"""
Zeus energy profiling integration for nanoGPT.

Wraps the Zeus ML energy monitoring library to track GPU energy consumption
during training. Falls back gracefully when Zeus is not installed or when
running on CPU.

Install: pip install zeus-ml
Docs:    https://ml.energy/zeus/
"""

_ZEUS_AVAILABLE = False
try:
    from zeus.monitor import ZeusMonitor
    _ZEUS_AVAILABLE = True
except ImportError:
    pass


def is_zeus_available() -> bool:
    """Return True if the zeus-ml package is installed and importable."""
    return _ZEUS_AVAILABLE


def create_zeus_monitor(gpu_indices=None):
    """
    Create a ZeusMonitor instance for the given GPU indices.

    Args:
        gpu_indices: list of GPU indices to monitor (e.g. [0]).
                     If None, monitors all visible GPUs.

    Returns:
        ZeusMonitor instance, or None if Zeus is unavailable.
    """
    if not _ZEUS_AVAILABLE:
        return None
    try:
        monitor = ZeusMonitor(gpu_indices=gpu_indices)
        return monitor
    except Exception as e:
        print(f"[energy_monitoring] Failed to create ZeusMonitor: {e}")
        return None


def begin_measurement(monitor, label="training"):
    """Start an energy measurement window."""
    if monitor is None:
        return
    try:
        monitor.begin_window(label)
    except Exception as e:
        print(f"[energy_monitoring] begin_window failed: {e}")


def end_measurement(monitor, label="training"):
    """
    End an energy measurement window and return total energy in joules.

    Returns:
        Total GPU energy consumed in joules, or 0.0 on failure.
    """
    if monitor is None:
        return 0.0
    try:
        measurement = monitor.end_window(label)
        return measurement.total_energy
    except Exception as e:
        print(f"[energy_monitoring] end_window failed: {e}")
        return 0.0
