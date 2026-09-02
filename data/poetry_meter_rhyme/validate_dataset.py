#!/usr/bin/env python3
"""Validate generated records, provenance, and cross-split isolation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(data_dir):
    hashes, groups = defaultdict(set), defaultdict(set)
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise ValueError(f"missing {path}")
        for number, row in enumerate(load(path), 1):
            for field in ("id", "group_id", "task", "source", "content_hash"):
                if not row.get(field):
                    raise ValueError(f"{path}:{number}: missing {field}")
            hashes[row["content_hash"]].add(split)
            groups[row["group_id"]].add(split)
            if split == "test" and (row.get("author") or row.get("title")):
                raise ValueError(f"{path}:{number}: benchmark exposes author/title")
    leaked_hashes = [key for key, values in hashes.items() if len(values) > 1]
    leaked_poets = [key for key, values in groups.items() if key.startswith("poet:") and len(values) > 1]
    if leaked_hashes or leaked_poets:
        raise ValueError(f"split leakage: {len(leaked_hashes)} content hashes, {len(leaked_poets)} poets")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if any(manifest["split_leakage"].values()):
        raise ValueError("manifest reports split leakage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    validate(args.data_dir)
    print("dataset validation passed")


if __name__ == "__main__":
    main()
