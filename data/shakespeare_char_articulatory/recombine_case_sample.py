#!/usr/bin/env python3
"""Split sample.py multicontext output and recombine lowercase + case lanes."""
from __future__ import annotations
import argparse
from pathlib import Path


def extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in text.splitlines():
        if line == "---------------":
            if cur is not None:
                sections[cur] = "\n".join(buf).strip("\n")
            cur, buf = None, []
            continue
        stripped = line.strip()
        if stripped.startswith("shakespeare_char_articulatory/") and stripped.endswith(":"):
            if cur is not None:
                sections[cur] = "\n".join(buf).strip("\n")
            cur = stripped[:-1]
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip("\n")
    return sections


def apply_case(lower: str, case: str) -> str:
    out = []
    for ch, marker in zip(lower, case):
        out.append(ch.upper() if marker == "u" else ch)
    return "".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sample_file")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    raw = Path(args.sample_file).read_text(encoding="utf-8")
    sections = extract_sections(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in sections.items():
        lane = name.rsplit("/", 1)[-1]
        path = out_dir / f"{lane}.txt"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {path}")
    lower = sections.get("shakespeare_char_articulatory/lowercase", "")
    case = sections.get("shakespeare_char_articulatory/case", "")
    final = apply_case(lower, case)
    final_path = out_dir / "recombined_lowercase_plus_case.txt"
    final_path.write_text(final + "\n", encoding="utf-8")
    print("\nRecombined lowercase + capitalization output:\n")
    print(final)
    print(f"\nwrote {final_path}")

if __name__ == "__main__":
    main()
