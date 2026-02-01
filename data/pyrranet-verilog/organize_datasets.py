#!/usr/bin/env python3
"""Download PyraNet-Verilog from Hugging Face and extract code samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the PyraNet-Verilog dataset and split each code sample into"
            " its own file named orig_<index>.v."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "orig",
        help="Directory to write orig_<index>.v files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for number of rows to extract.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming (downloads the entire dataset).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    return parser.parse_args()


def write_samples(output_dir: Path, max_rows: int | None, streaming: bool, overwrite: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("bnadimi/PyraNet-Verilog", split="train", streaming=streaming)

    count = 0
    for row in dataset:
        filename = output_dir / f"orig_{count:07d}.v"
        if filename.exists() and not overwrite:
            count += 1
            if max_rows is not None and count >= max_rows:
                break
            continue

        code = row.get("code")
        if code is None:
            count += 1
            if max_rows is not None and count >= max_rows:
                break
            continue

        filename.write_text(code.strip() + "\n", encoding="utf-8")
        count += 1
        if max_rows is not None and count >= max_rows:
            break

    return count


def main() -> None:
    args = parse_args()
    total = write_samples(
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        streaming=not args.no_streaming,
        overwrite=args.overwrite,
    )
    print(f"Wrote {total} files to {args.output_dir}")


if __name__ == "__main__":
    main()
