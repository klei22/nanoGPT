#!/usr/bin/env python3
"""Split python fenced code blocks into separate files for tokenizer training."""

import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split ```python fenced blocks into files")
    parser.add_argument("--input", required=True, help="Path to the text file with output field contents")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted Python files")
    parser.add_argument("--prefix", default="python_snippet_", help="Prefix for emitted filenames")
    parser.add_argument("--start_index", type=int, default=1, help="Starting index for numbered files")
    parser.add_argument("--extension", default=".py", help="Extension for emitted files")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    instruction_re = re.compile(r'"""<start>(.*?)"""', re.DOTALL | re.IGNORECASE)
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

        # kind == "code"
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


if __name__ == "__main__":
    main()
