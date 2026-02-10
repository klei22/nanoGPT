#!/usr/bin/env python3
"""Barrel shift .bin token files.

Supports shifting a single file or all .bin files in a directory. When operating
on a directory, files are first copied to a new directory, then shifted in-place.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barrel shift token .bin files.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Path to a single .bin file to shift.",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing .bin files to shift.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help=(
            "Directory to write shifted files. Required when using --input_dir. "
            "If --num_variants > 1, variant subdirectories are created here."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help=(
            "Output file path for single-file mode. If omitted, --output_dir "
            "must be provided and the input filename is reused."
        ),
    )
    parser.add_argument(
        "--shift_amount",
        type=int,
        default=1,
        help="Number of positions to barrel shift.",
    )
    parser.add_argument(
        "--shift_direction",
        choices=["f", "b"],
        default="b",
        help=(
            "Shift direction. 'f' moves the last token to the front (indices move forward). "
            "'b' moves the first token to the end (indices move backward)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="uint16",
        choices=["uint16", "uint32", "int32", "int64"],
        help="Token dtype stored in the .bin files.",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=1,
        help=(
            "Number of shifted dataset variants to create. When > 1, creates "
            "directories named b_<n>_<shift>_<f|b> (n is 0-indexed)."
        ),
    )
    return parser.parse_args()


def _normalize_shift(shift_amount: int, length: int) -> int:
    if length == 0:
        return 0
    return shift_amount % length


def _barrel_shift_array(data: np.ndarray, shift_amount: int, direction: str) -> np.ndarray:
    if data.size == 0:
        return data
    shift_amount = _normalize_shift(shift_amount, data.size)
    if shift_amount == 0:
        return data.copy()
    shift = shift_amount if direction == "f" else -shift_amount
    return np.roll(data, shift)


def shift_file(input_file: Path, output_file: Path, shift_amount: int, direction: str, dtype: np.dtype) -> None:
    data = np.fromfile(input_file, dtype=dtype)
    shifted = _barrel_shift_array(data, shift_amount, direction)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shifted.tofile(output_file)


def _copy_dir(source_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        raise FileExistsError(f"Destination already exists: {dest_dir}")
    shutil.copytree(source_dir, dest_dir)


def shift_directory(source_dir: Path, dest_dir: Path, shift_amount: int, direction: str, dtype: np.dtype) -> None:
    _copy_dir(source_dir, dest_dir)
    for root, _, files in os.walk(dest_dir):
        for filename in sorted(files):
            if not filename.endswith(".bin"):
                continue
            file_path = Path(root) / filename
            shift_file(file_path, file_path, shift_amount, direction, dtype)


def _variant_shift_amount(base_shift: int, variant_index: int) -> int:
    return base_shift + variant_index


def main() -> None:
    args = parse_args()

    dtype = np.dtype(args.dtype)
    num_variants = max(args.num_variants, 1)
    direction = args.shift_direction

    if args.input_dir:
        if args.output_dir is None:
            raise ValueError("--output_dir is required when using --input_dir.")
        source_dir = Path(args.input_dir)
        dest_root = Path(args.output_dir)
        if num_variants > 1:
            dest_root.mkdir(parents=True, exist_ok=True)
        else:
            dest_root.parent.mkdir(parents=True, exist_ok=True)

        for idx in range(num_variants):
            shift_amount = _variant_shift_amount(args.shift_amount, idx)
            if num_variants > 1:
                variant_dir = dest_root / f"b_{idx}_{shift_amount}_{direction}"
            else:
                variant_dir = dest_root
            shift_directory(source_dir, variant_dir, shift_amount, direction, dtype)
    else:
        input_file = Path(args.input_file)
        if args.output_file:
            output_file = Path(args.output_file)
        elif args.output_dir:
            output_file = Path(args.output_dir) / input_file.name
        else:
            raise ValueError("Provide --output_file or --output_dir for single-file mode.")

        if num_variants > 1 and args.output_dir:
            dest_root = Path(args.output_dir)
            dest_root.mkdir(parents=True, exist_ok=True)
            for idx in range(num_variants):
                shift_amount = _variant_shift_amount(args.shift_amount, idx)
                variant_dir = dest_root / f"b_{idx}_{shift_amount}_{direction}"
                variant_dir.mkdir(parents=True, exist_ok=True)
                variant_file = variant_dir / input_file.name
                shift_file(input_file, variant_file, shift_amount, direction, dtype)
        else:
            shift_file(input_file, output_file, args.shift_amount, direction, dtype)


if __name__ == "__main__":
    main()
