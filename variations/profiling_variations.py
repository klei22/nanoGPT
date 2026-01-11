"""Profiling presets for sample.py inference runs.

Each entry in ``profiling_variations`` defines the torch.profiler configuration
used when ``--profile_variation`` is provided to ``sample.py``.
"""
from __future__ import annotations

from torch.profiler import ProfilerActivity

profiling_variations = {
    "cpu_latency": {
        "activities": [ProfilerActivity.CPU],
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": True,
        "sort_by": "self_cpu_time_total",
        "row_limit": 20,
        "trace_prefix": "cpu_latency",
    },
    "cuda_kernels": {
        "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
        "with_modules": True,
        "sort_by": "self_cuda_time_total",
        "row_limit": 30,
        "trace_prefix": "cuda_kernels",
    },
    "flops_and_io": {
        "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        "record_shapes": False,
        "profile_memory": True,
        "with_stack": True,
        "with_flops": True,
        "sort_by": "self_cuda_time_total",
        "row_limit": 30,
        "trace_prefix": "flops_io",
    },
}
