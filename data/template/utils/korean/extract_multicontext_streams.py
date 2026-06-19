#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

try:
    import yaml
except ImportError:
    yaml = None

from hangul_factorizer import HangulFactorizedTokenizer, dump_json


def main() -> None:
    p = argparse.ArgumentParser(description="Split Korean text into aligned 23-lane Hangul factor streams.")
    p.add_argument("input", help="UTF-8 Korean text file")
    p.add_argument("output_dir", help="Directory that will receive lane/input.txt files")
    p.add_argument("--metadata-json", default="metadata.json")
    p.add_argument("--metadata-yaml", default="metadata.yaml")
    args = p.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = HangulFactorizedTokenizer()

    lane_chars = [[] for _ in tok.lane_names]
    records = []
    for pos, ch in enumerate(text):
        ids = tok.encode_char(ch)
        records.append(tok.metadata_for_char(ch, pos))
        for i, idx in enumerate(ids):
            lane_chars[i].append(tok.token_for(i, idx))

    for name, chars in zip(tok.lane_names, lane_chars):
        lane_dir = out / name
        lane_dir.mkdir(parents=True, exist_ok=True)
        (lane_dir / "input.txt").write_text("".join(chars), encoding="utf-8")

    char_dir = out / "char"
    char_dir.mkdir(parents=True, exist_ok=True)
    (char_dir / "input.txt").write_text(text, encoding="utf-8")

    metadata = {"lanes": tok.lane_metadata(), "characters": records}
    dump_json(out / args.metadata_json, metadata)
    if yaml is not None:
        (out / args.metadata_yaml).write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        (out / args.metadata_yaml).write_text("# Install PyYAML for native YAML output. JSON-compatible metadata follows.\n" + (out / args.metadata_json).read_text(encoding="utf-8"), encoding="utf-8")

if __name__ == "__main__":
    main()
