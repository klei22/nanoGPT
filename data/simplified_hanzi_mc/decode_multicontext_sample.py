#!/usr/bin/env python3
"""Reconstruct simplified Hanzi from generated multicontext lane text.

The `char` lane carries simplified Hanzi. For timesteps where `char` is `⧆`,
the `non_hanzi` lane carries the original U+XXXX code point, allowing full-text
reconstruction instead of rendering `<NON_HANZI>` placeholders.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
NON_HANZI="⧆"
PLACEHOLDER="∅"
def decode_non_hanzi_value(value: str) -> str:
    if value == PLACEHOLDER:
        return PLACEHOLDER
    if not value.startswith("U+"):
        raise ValueError(f"Expected U+XXXX non_hanzi value, got {value!r}")
    return chr(int(value[2:], 16))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="data/simplified_hanzi_mc")
    ap.add_argument("--char_file", default=None, help="Optional generated char-lane text file; defaults to <root>/char/input.txt")
    ap.add_argument("--non_hanzi_file", default=None, help="Optional generated non_hanzi-lane text file; defaults to <root>/non_hanzi/input.txt")
    args=ap.parse_args()
    root=Path(args.root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    char_path=Path(args.char_file) if args.char_file else root/"char"/"input.txt"
    non_hanzi_path=Path(args.non_hanzi_file) if args.non_hanzi_file else root/"non_hanzi"/"input.txt"
    chars=[line.rstrip("\n") for line in char_path.read_text(encoding="utf-8").splitlines()]
    non_hanzi=[decode_non_hanzi_value(line.rstrip("\n")) for line in non_hanzi_path.read_text(encoding="utf-8").splitlines()]
    if len(chars) != len(non_hanzi):
        raise ValueError(f"Lane length mismatch: char={len(chars)} non_hanzi={len(non_hanzi)}")
    decoded=[nh if ch==NON_HANZI else ch for ch, nh in zip(chars, non_hanzi)]
    print("".join(decoded))
    print(f"decoded_steps={len(decoded)} lanes={','.join(manifest['lanes'])}")
if __name__ == "__main__": main()
