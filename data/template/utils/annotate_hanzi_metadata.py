"""Annotate JSON datasets with Hanzi metadata.

This utility reads a JSON array (or JSONL) dataset and adds an additional
key to each entry containing Chinese-centric metadata such as radicals,
pinyin, definitions, and frequency. The script is intended to be run on
JSON files downloaded via ``get_dataset.sh`` before tokenization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Sequence

try:
    from hanzipy.decomposer import HanziDecomposer
    from hanzipy.dictionary import HanziDictionary
except ImportError as exc:  # pragma: no cover - dependency handled at runtime
    raise SystemExit(
        "hanzipy is required for annotate_hanzi_metadata.py.\n"
        "Install it with `pip install hanzipy`."
    ) from exc

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array when using bracketed JSON input")
        return data
    # Fallback: JSONL format
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def chinese_characters(text: str) -> Iterable[str]:
    for char in text:
        if CJK_PATTERN.match(char):
            yield char


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def build_metadata_for_char(
    char: str,
    decomposer: HanziDecomposer,
    dictionary: HanziDictionary,
    metadata_types: Sequence[str],
) -> Dict:
    info: Dict[str, object] = {}

    decomposition: Dict[str, object] | None = None
    if "radical" in metadata_types or "components" in metadata_types:
        # Request the full decomposition tree so downstream consumers can pick
        # the level they want (once/radical/graphical).
        decomposition = decomposer.decompose(char)
        if "radical" in metadata_types:
            info["radical"] = decomposition.get("radical") or []
        if "components" in metadata_types:
            info["components"] = decomposition
    if "pinyin" in metadata_types:
        info["pinyin"] = dictionary.get_pinyin(char)
    if "definitions" in metadata_types:
        info["definitions"] = dictionary.definition_lookup(char)
    if "frequency" in metadata_types:
        info["frequency"] = dictionary.get_character_frequency(char)
    if "phonetic_regularity" in metadata_types:
        info["phonetic_regularity"] = dictionary.determine_phonetic_regularity(char)
    return info


def annotate_entries(
    entries: List[Dict],
    text_key: str,
    metadata_key: str,
    metadata_types: Sequence[str],
) -> List[Dict]:
    decomposer = HanziDecomposer()
    dictionary = HanziDictionary()

    for entry in entries:
        text = entry.get(text_key)
        if not isinstance(text, str):
            continue
        chars = unique_preserve_order(chinese_characters(text))
        char_metadata = {
            char: build_metadata_for_char(char, decomposer, dictionary, metadata_types)
            for char in chars
        }
        entry[metadata_key] = char_metadata
    return entries


def write_dataset(entries: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a JSON dataset with Chinese metadata (radicals, pinyin, etc.) "
            "using the hanzipy library."
        )
    )
    parser.add_argument("--input_json", required=True, help="Path to source JSON or JSONL dataset.")
    parser.add_argument("--output_json", required=True, help="Path to write the annotated JSON array.")
    parser.add_argument(
        "--text_key",
        default="text",
        help="Key in each entry containing the text to inspect for Chinese characters.",
    )
    parser.add_argument(
        "--metadata_key",
        default="hanzi_metadata",
        help="Key to store the generated metadata alongside each entry.",
    )
    parser.add_argument(
        "--metadata_types",
        nargs="+",
        default=["radical", "pinyin"],
        choices=[
            "radical",
            "components",
            "pinyin",
            "definitions",
            "frequency",
            "phonetic_regularity",
        ],
        help="Metadata types to attach to each Chinese character encountered.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.input_json)
    annotated = annotate_entries(dataset, args.text_key, args.metadata_key, args.metadata_types)
    write_dataset(annotated, args.output_json)
    print(f"Annotated {len(annotated)} entries -> {args.output_json}")
