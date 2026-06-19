"""Utility for generating chained arithmetic operation datasets."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence


AVAILABLE_OPERATIONS = {
    "addition": "addition",
    "multiplication": "multiplication",
    "shift_right": "shift_right",
    "shift_left": "shift_left",
    "reverse": "reverse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic dataset consisting of chained arithmetic "
            "operations."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("input.txt"),
        help="Path to the output text file (default: input.txt).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="Number of independent chains to generate (default: 1000).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help="Number of operations to apply per chain (default: 16).",
    )
    parser.add_argument(
        "--operations",
        type=str,
        default=",".join(AVAILABLE_OPERATIONS),
        help=(
            "Comma separated list of operations to sample from. Available "
            "choices: addition, multiplication, shift_right, shift_left, "
            "reverse."
        ),
    )
    parser.add_argument(
        "--modulo",
        type=int,
        default=1000,
        help="Modulo applied after every operation. Use -1 to disable.",
    )
    parser.add_argument(
        "--initial-value",
        type=int,
        default=0,
        help="Starting value for each chain when --random-initial is not set.",
    )
    parser.add_argument(
        "--random-initial",
        action="store_true",
        help="Sample the initial value for each chain uniformly at random.",
    )
    parser.add_argument(
        "--add-min",
        type=int,
        default=0,
        help="Minimum operand used for addition (inclusive).",
    )
    parser.add_argument(
        "--add-max",
        type=int,
        default=999,
        help="Maximum operand used for addition (inclusive).",
    )
    parser.add_argument(
        "--mul-min",
        type=int,
        default=0,
        help="Minimum operand used for multiplication (inclusive).",
    )
    parser.add_argument(
        "--mul-max",
        type=int,
        default=999,
        help="Maximum operand used for multiplication (inclusive).",
    )
    parser.add_argument(
        "--format",
        choices=("stacked", "inline"),
        default="stacked",
        help=(
            "Output format. 'stacked' alternates values and operations over "
            "multiple lines, while 'inline' emits expressions like 'a+b=c'."
        ),
    )
    parser.add_argument(
        "--no-separators",
        action="store_true",
        help="Disable blank lines between independent chains.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def validate_range(name: str, minimum: int, maximum: int) -> None:
    if minimum > maximum:
        raise ValueError(f"Invalid range for {name}: {minimum} > {maximum}")


def parse_operation_list(operation_list: str) -> List[str]:
    selections = [item.strip() for item in operation_list.split(",") if item.strip()]
    if not selections:
        raise ValueError("At least one operation must be specified.")
    for op in selections:
        if op not in AVAILABLE_OPERATIONS:
            raise ValueError(f"Unsupported operation: {op}")
    return selections


def apply_modulo(value: int, modulo: int) -> int:
    if modulo is not None and modulo > 0:
        return value % modulo
    return value


def rotate_digits(value: int, direction: str) -> int:
    if value == 0:
        return 0
    sign = -1 if value < 0 else 1
    digits = list(str(abs(value)))
    if len(digits) == 1:
        return value
    if direction == "left":
        rotated = digits[1:] + digits[:1]
    elif direction == "right":
        rotated = digits[-1:] + digits[:-1]
    else:
        raise ValueError(f"Unsupported rotation direction: {direction}")
    return sign * int("".join(rotated))


def reverse_digits(value: int) -> int:
    sign = -1 if value < 0 else 1
    digits = str(abs(value))
    return sign * int(digits[::-1]) if digits else 0


def sample_initial_value(rng: random.Random, args: argparse.Namespace) -> int:
    if not args.random_initial:
        value = args.initial_value
    else:
        if args.modulo and args.modulo > 0:
            value = rng.randrange(args.modulo)
        else:
            high = max(args.add_max, args.mul_max, 1)
            value = rng.randint(0, high)
    return apply_modulo(value, args.modulo)


def perform_operation(
    op_name: str, current: int, rng: random.Random, args: argparse.Namespace
) -> tuple[str, str, int]:
    if op_name == "addition":
        operand = rng.randint(args.add_min, args.add_max)
        result = current + operand
        op_token = f"+{operand}"
        inline_repr = f"{current}+{operand}"
    elif op_name == "multiplication":
        operand = rng.randint(args.mul_min, args.mul_max)
        result = current * operand
        op_token = f"*{operand}"
        inline_repr = f"{current}*{operand}"
    elif op_name == "shift_right":
        result = rotate_digits(current, "right")
        op_token = "R"
        inline_repr = f"{current}R"
    elif op_name == "shift_left":
        result = rotate_digits(current, "left")
        op_token = "L"
        inline_repr = f"{current}L"
    elif op_name == "reverse":
        result = reverse_digits(current)
        op_token = "r"
        inline_repr = f"{current}r"
    else:
        raise ValueError(f"Unsupported operation: {op_name}")

    result = apply_modulo(result, args.modulo)
    return op_token, inline_repr, result


def format_chain(
    sequence: Sequence[tuple[str, str, int]],
    starting_value: int,
    output_format: str,
) -> List[str]:
    if output_format == "stacked":
        lines: List[str] = [str(starting_value)]
        current = starting_value
        for op_token, _inline_expr, result in sequence:
            lines.append(op_token)
            lines.append(str(result))
            current = result
        return lines
    elif output_format == "inline":
        return [f"{inline_expr}={result}" for _, inline_expr, result in sequence]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def generate_dataset(args: argparse.Namespace) -> Iterable[str]:
    if args.seed is not None:
        rng = random.Random(args.seed)
    else:
        rng = random.Random()

    operations = parse_operation_list(args.operations)
    validate_range("addition", args.add_min, args.add_max)
    validate_range("multiplication", args.mul_min, args.mul_max)

    for _ in range(args.num_sequences):
        starting_value = sample_initial_value(rng, args)
        sequence = []
        current = starting_value
        for _ in range(args.num_steps):
            op_name = rng.choice(operations)
            op_token, inline_repr, current = perform_operation(
                op_name, current, rng, args
            )
            sequence.append((op_token, inline_repr, current))
        for line in format_chain(sequence, starting_value, args.format):
            yield line
        if not args.no_separators:
            yield ""


def main() -> None:
    args = parse_args()
    output_lines = list(generate_dataset(args))
    if output_lines and output_lines[-1] == "":
        output_lines = output_lines[:-1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
