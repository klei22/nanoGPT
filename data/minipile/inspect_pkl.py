#!/usr/bin/env python3
"""Inspect the top-level structure of a pickle / torch checkpoint file.

Usage:
  python inspect_pickle.py path/to/file.pkl

It will attempt torch.load (CPU map) if torch is available, otherwise fall back
to the stdlib pickle loader. For large objects, it prints a shallow summary with
key names, lengths, and shapes/dtypes where available.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import numpy as np  # type: ignore
    _NP_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    _NP_AVAILABLE = False


def _describe_tensor(x: Any) -> str:
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
    if _NP_AVAILABLE and isinstance(x, np.ndarray):
        return f"ndarray(shape={x.shape}, dtype={x.dtype})"
    return type(x).__name__


def _summarize(obj: Any, depth: int = 0, max_items: int = 10) -> str:
    indent = "  " * depth
    prefix = indent + "- "

    # Simple scalars
    if isinstance(obj, (str, int, float, bool, type(None))):
        return f"{prefix}{repr(obj)} ({type(obj).__name__})"

    # torch / numpy tensors
    if (_TORCH_AVAILABLE and isinstance(obj, torch.Tensor)) or (_NP_AVAILABLE and isinstance(obj, np.ndarray)):
        return f"{prefix}{_describe_tensor(obj)}"

    # Mappings (dict-like)
    if isinstance(obj, Mapping):
        lines = [f"{prefix}{type(obj).__name__} with {len(obj)} keys:"]
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                lines.append(indent + "  ...")
                break
            lines.append(f"{indent}  key={k!r}: {_describe_tensor(v) if (isinstance(v, (list, tuple, Mapping)) is False) else type(v).__name__}")
        return "\n".join(lines)

    # Sequences
    if isinstance(obj, (list, tuple)):
        lines = [f"{prefix}{type(obj).__name__} len={len(obj)}"]
        for i, v in enumerate(obj):
            if i >= max_items:
                lines.append(indent + "  ...")
                break
            lines.append(f"{indent}  [{i}]: {type(v).__name__}")
        return "\n".join(lines)

    # Fallback
    return f"{prefix}{_describe_tensor(obj)}"


def load_any(path: Path) -> Any:
    if _TORCH_AVAILABLE:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            pass
    # Fallback to pickle
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a pickle / torch checkpoint file")
    parser.add_argument("path", type=Path, help="Path to the .pkl / .pt / .pth file")
    parser.add_argument("--max-items", type=int, default=10, help="Max items to show from mappings/sequences (default: 10)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    try:
        obj = load_any(args.path)
    except Exception as exc:
        print(f"Failed to load {args.path}: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded: {type(obj).__name__}")
    print(_summarize(obj, depth=0, max_items=args.max_items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
