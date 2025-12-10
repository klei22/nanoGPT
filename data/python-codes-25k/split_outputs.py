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

    pattern = re.compile(r"```python\s+(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(content)

    index = args.start_index
    for block in matches:
        filename = f"{args.prefix}{index:05d}{args.extension}"
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as snippet_file:
            snippet_file.write(block.strip() + "\n")
        index += 1

    print(f"Wrote {index - args.start_index} files to {output_dir}")


if __name__ == "__main__":
    main()
