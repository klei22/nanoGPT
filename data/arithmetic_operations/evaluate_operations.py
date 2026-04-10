"""Evaluate the formatting quality of arithmetic operation outputs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple

STACKED_NUMBER_PATTERN = re.compile(r"^-?\d+$")
STACKED_OPERATION_PATTERN = re.compile(r"^(?:[+*]-?\d+|[RLr])$")
INLINE_BINARY_PATTERN = re.compile(
    r"^\s*-?\d+\s*[+*]\s*-?\d+\s*=\s*-?\d+\s*$"
)
INLINE_UNARY_PATTERN = re.compile(r"^\s*-?\d+\s*[RLr]\s*=\s*-?\d+\s*$")


FORMATS = ("auto", "stacked", "inline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how many lines in a generated sample conform to the "
            "expected arithmetic operation formats."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the text file produced by sample.py or train.py",
    )
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="auto",
        help=(
            "Expected formatting style: 'stacked', 'inline', or 'auto' to "
            "infer based on the data."
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Count empty lines as invalid entries in the statistics.",
    )
    return parser.parse_args()


def detect_format(lines: Iterable[str]) -> str:
    has_equals = any("=" in line for line in lines)
    return "inline" if has_equals else "stacked"


def is_valid_line(line: str, fmt: str) -> bool:
    if fmt == "stacked":
        return bool(STACKED_NUMBER_PATTERN.match(line)) or bool(
            STACKED_OPERATION_PATTERN.match(line)
        )
    if fmt == "inline":
        return bool(INLINE_BINARY_PATTERN.match(line) or INLINE_UNARY_PATTERN.match(line))
    raise ValueError(f"Unsupported format: {fmt}")


def evaluate(lines: Iterable[str], fmt: str, include_empty: bool) -> Tuple[int, int]:
    total = 0
    valid = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line and not include_empty:
            continue
        total += 1
        if line and is_valid_line(line, fmt):
            valid += 1
        elif not line and include_empty:
            # Empty lines only count toward totals when explicitly requested.
            pass
    return total, valid


def render_table(total: int, valid: int) -> str:
    invalid = total - valid
    fraction = (valid / total) if total else 0.0
    rows = [
        ("Total lines", f"{total}"),
        ("Valid lines", f"{valid}"),
        ("Invalid lines", f"{invalid}"),
        ("Valid fraction", f"{fraction:.4f}"),
    ]
    header = "+----------------+-----------+"
    table_lines = [header, "| Metric         | Value     |", header]
    for metric, value in rows:
        table_lines.append(f"| {metric:<14} | {value:<9} |")
    table_lines.append(header)
    return "\n".join(table_lines)


def main() -> None:
    args = parse_args()
    raw_lines = args.input.read_text().splitlines()
    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(raw_lines)
    total, valid = evaluate(raw_lines, fmt, args.include_empty)
    table = render_table(total, valid)
    print(table)


if __name__ == "__main__":
    main()
