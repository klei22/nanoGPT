#!/usr/bin/env python3
"""Combine outputs and selected annotations when character counts match."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate outputs/general/param_nesting lengths and emit combined artifacts."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        required=True,
        help="Directory containing emitted .py snippets.",
    )
    parser.add_argument(
        "--json-out",
        help="Path for the combined JSON array (default: matched_annotations.json next to outputs dir)",
    )
    parser.add_argument(
        "--concat-out",
        help="Path for the concatenated text output (default: inputs_concat.txt next to outputs dir)",
    )
    parser.add_argument(
        "--mc-out-dir",
        help=(
            "Base directory for mc_* folders (default: alongside outputs dir). "
            "mc_out, mc_pna, mc_ga will be created inside."
        ),
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def collect_entries(
    outputs_dir: Path, general_dir: Path, param_dir: Path
) -> Tuple[List[dict], int, int, int]:
    entries = []
    missing = 0
    mismatched = 0
    processed = 0

    for py_file in sorted(outputs_dir.glob("*.py")):
        processed += 1
        stem = py_file.stem
        general_path = general_dir / f"{stem}.mapped"
        param_path = param_dir / f"{stem}.mapped"

        if not general_path.exists() or not param_path.exists():
            missing += 1
            continue

        output_text = read_text(py_file)
        general_text = read_text(general_path)
        param_text = read_text(param_path)

        if len(output_text) == len(general_text) == len(param_text):
            entries.append(
                {
                    "name": stem,
                    "output": output_text,
                    "general": general_text,
                    "param_nesting": param_text,
                }
            )
        else:
            mismatched += 1

    return entries, processed, missing, mismatched


def write_json(entries: List[dict], path: Path) -> None:
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_concat(entries: List[dict], path: Path) -> None:
    chunks: List[str] = []
    for entry in entries:
        chunks.extend([entry["output"], entry["param_nesting"], entry["general"]])
    path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def write_mc_inputs(entries: List[dict], base_dir: Path) -> None:
    mc_out = base_dir / "mc_out"
    mc_pna = base_dir / "mc_pna"
    mc_ga = base_dir / "mc_ga"

    for folder in (mc_out, mc_pna, mc_ga):
        folder.mkdir(parents=True, exist_ok=True)

    out_path = mc_out / "input.txt"
    pna_path = mc_pna / "input.txt"
    ga_path = mc_ga / "input.txt"

    out_chunks: List[str] = []
    pna_chunks: List[str] = []
    ga_chunks: List[str] = []

    for entry in entries:
        out_chunks.append(entry["output"])
        pna_chunks.append(entry["param_nesting"])
        ga_chunks.append(entry["general"])

    out_path.write_text("out:\n" + "\n\n".join(out_chunks).rstrip() + "\n\n", encoding="utf-8")
    pna_path.write_text("pna:\n" + "\n\n".join(pna_chunks).rstrip() + "\n\n", encoding="utf-8")
    ga_path.write_text("gen:\n" + "\n\n".join(ga_chunks).rstrip() + "\n\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    parent_dir = outputs_dir.parent

    general_dir = parent_dir / "general_annotations"
    param_dir = parent_dir / "param_nesting_annotations"

    json_out = Path(args.json_out) if args.json_out else parent_dir / "matched_annotations.json"
    concat_out = Path(args.concat_out) if args.concat_out else parent_dir / "inputs_concat.txt"
    mc_base = Path(args.mc_out_dir) if args.mc_out_dir else parent_dir

    entries, processed, missing, mismatched = collect_entries(outputs_dir, general_dir, param_dir)

    if entries:
        write_json(entries, json_out)
        write_concat(entries, concat_out)
        write_mc_inputs(entries, mc_base)

    print(
        "Processed: "
        f"{processed} files. "
        f"Collected: {len(entries)} matching entries. "
        f"Missing annotations: {missing}. "
        f"Length mismatches: {mismatched}."
    )
    if not entries:
        print("No matching entries were written.")


if __name__ == "__main__":
    main()
