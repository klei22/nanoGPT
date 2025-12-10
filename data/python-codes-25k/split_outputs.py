#!/usr/bin/env python3
"""Split python fenced code blocks into separate files for tokenizer training."""

import argparse
import py_compile
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split ```python fenced blocks into files")
    parser.add_argument("--input", required=True, help="Path to the text file with output field contents")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted Python files")
    parser.add_argument("--prefix", default="python_snippet_", help="Prefix for emitted filenames")
    parser.add_argument("--start_index", type=int, default=1, help="Starting index for numbered files")
    parser.add_argument("--extension", default=".py", help="Extension for emitted files")
    parser.add_argument(
        "--format",
        action="store_true",
        help="Run `ruff format` on emitted files before compilation checks",
    )
    parser.add_argument(
        "--ruff-exec",
        default="ruff",
        help="Ruff executable to use when formatting (default: ruff)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip post-processing compilation checks",
    )
    return parser.parse_args()


def run_formatter(output_dir: Path, ruff_exec: str) -> None:
    """Attempt to format all emitted files with ruff."""
    try:
        result = subprocess.run(
            [ruff_exec, "format", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
    except FileNotFoundError:
        print(f"Formatter '{ruff_exec}' not found; skipping formatting step.")
    except subprocess.CalledProcessError as exc:
        print("ruff format failed; continuing without aborting.")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)


def compile_snippets(output_dir: Path, extension: str) -> None:
    """Compile emitted snippets, moving failures to a quarantine folder."""
    fail_dir = output_dir / "does_not_compile"
    compiled = 0
    failed = 0

    for path in sorted(output_dir.glob(f"*{extension}")):
        if not path.is_file():
            continue

        try:
            py_compile.compile(str(path), doraise=True)
            compiled += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            fail_dir.mkdir(parents=True, exist_ok=True)
            target = fail_dir / path.name
            print(f"Compile failed for {path.name}: {exc}. Moving to {fail_dir}.")
            path.replace(target)

    summary = [f"{compiled} compiled successfully"]
    if failed:
        summary.append(f"{failed} moved to {fail_dir}")
    print("Compilation summary: " + ", ".join(summary))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    instruction_re = re.compile(r'"""(?:<start>|start)(.*?)"""', re.DOTALL | re.IGNORECASE)
    code_re = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    events = []
    events.extend((m.start(), "instruction", m.group(1)) for m in instruction_re.finditer(content))
    events.extend((m.start(), "code", m.group(1)) for m in code_re.finditer(content))
    events.sort(key=lambda e: e[0])

    index = args.start_index
    pending_instruction = None

    for _, kind, payload in events:
        if kind == "instruction":
            pending_instruction = payload
            continue

        code_block = payload.strip()
        instruction_block = pending_instruction.strip() if pending_instruction is not None else None
        pending_instruction = None

        snippet_parts = []
        if instruction_block:
            snippet_parts.append('"""')
            snippet_parts.append(instruction_block)
            snippet_parts.append('"""\n')
        snippet_parts.append(code_block)

        filename = f"{args.prefix}{index:05d}{args.extension}"
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as snippet_file:
            snippet_file.write("\n".join(snippet_parts).rstrip() + "\n")
        index += 1

    print(f"Wrote {index - args.start_index} files to {output_dir}")

    if args.format:
        run_formatter(output_dir, args.ruff_exec)

    if not args.skip_compile:
        compile_snippets(output_dir, args.extension)


if __name__ == "__main__":
    main()
