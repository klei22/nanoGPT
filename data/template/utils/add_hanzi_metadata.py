"""Annotate JSON records with hanzipy metadata.

This script augments JSON or JSONL datasets with per-character metadata
from `hanzipy`, such as pinyin or radical information. It is meant to be
used after fetching datasets with ``get_dataset.sh`` when the downloaded
JSON contains Chinese text entries that need extra linguistic context.

Example:
    python add_hanzi_metadata.py \
        --input_json downloaded.json \
        --text_key text \
        --include_pinyin \
        --include_radicals \
        --output_json downloaded_with_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from hanzipy.decomposer import HanziDecomposer
from hanzipy.dictionary import HanziDictionary

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append hanzipy metadata (pinyin, radicals, etc.) to JSON records",
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to the JSON or JSONL file to annotate.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Where to write the annotated JSON. Defaults to the input name with '_annotated' suffix.",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="Key in each record that contains the text to analyze.",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default="hanzipy",
        help="Key name to store the generated metadata inside each record.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Treat the input as JSON Lines (one JSON object per line).",
    )
    parser.add_argument(
        "--include_pinyin",
        action="store_true",
        help="Annotate with possible pinyin readings for each character.",
    )
    parser.add_argument(
        "--include_radicals",
        action="store_true",
        help="Annotate with radical decompositions and their meanings.",
    )
    parser.add_argument(
        "--include_frequency",
        action="store_true",
        help="Annotate with character frequency information from hanzipy.",
    )
    parser.add_argument(
        "--include_decomposition",
        action="store_true",
        help="Annotate with full decomposition (once, radical, graphical).",
    )
    parser.add_argument(
        "--deduplicate_radicals",
        action="store_true",
        help="Store a set of unique radicals for the entry in addition to per-character data.",
    )
    return parser.parse_args()


def load_records(path: Path, jsonl: bool) -> List[Dict]:
    if jsonl:
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        LOGGER.info("Loaded %d JSONL rows from %s", len(records), path)
        return records

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        LOGGER.warning("Input JSON is an object; wrapping it in a list for processing.")
        return [data]
    if not isinstance(data, list):
        raise ValueError("Expected a list or dict at the top-level of the JSON file.")
    LOGGER.info("Loaded %d JSON objects from %s", len(data), path)
    return data


def save_records(records: Iterable[Dict], path: Path, jsonl: bool) -> None:
    with path.open("w", encoding="utf-8") as fp:
        if jsonl:
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False))
                fp.write("\n")
            return
        json.dump(records, fp, ensure_ascii=False, indent=2)


def get_radical_info(
    decomposer: HanziDecomposer, character: str, components: List[str]
) -> List[Dict[str, str]]:
    radical_info: List[Dict[str, str]] = []
    for radical in components:
        meaning = decomposer.get_radical_meaning(radical)
        if meaning:
            radical_info.append({"radical": radical, "meaning": meaning})
        else:
            radical_info.append({"radical": radical})
    return radical_info


def build_character_metadata(
    dictionary: HanziDictionary,
    decomposer: HanziDecomposer,
    character: str,
    include_pinyin: bool,
    include_radicals: bool,
    include_frequency: bool,
    include_decomposition: bool,
) -> Optional[Dict]:
    metadata: Dict[str, object] = {"character": character}

    if include_pinyin:
        pinyin_values = dictionary.get_pinyin(character)
        if pinyin_values:
            metadata["pinyin"] = pinyin_values

    radical_components: List[str] = []
    if include_radicals:
        decomposition = decomposer.decompose(character, 2)
        radical_components = decomposition.get("components") or decomposition.get("radical", [])
        if radical_components:
            metadata["radicals"] = radical_components
            metadata["radical_meanings"] = get_radical_info(decomposer, character, radical_components)

    if include_frequency:
        frequency = dictionary.get_character_frequency(character)
        if frequency:
            metadata["frequency"] = frequency

    if include_decomposition:
        metadata["decomposition"] = decomposer.decompose(character)

    if len(metadata) == 1:  # only the character itself
        return None
    return metadata


def annotate_record(
    record: Dict,
    dictionary: HanziDictionary,
    decomposer: HanziDecomposer,
    text_key: str,
    output_key: str,
    include_pinyin: bool,
    include_radicals: bool,
    include_frequency: bool,
    include_decomposition: bool,
    deduplicate_radicals: bool,
) -> Dict:
    content = record.get(text_key)
    if not isinstance(content, str):
        LOGGER.debug("Record missing text key '%s'; skipping metadata", text_key)
        return record

    characters = [c for c in content if not c.isspace()]
    char_metadata: List[Dict] = []
    unique_radicals: set[str] = set()

    for char in characters:
        per_char = build_character_metadata(
            dictionary,
            decomposer,
            char,
            include_pinyin,
            include_radicals,
            include_frequency,
            include_decomposition,
        )
        if per_char:
            char_metadata.append(per_char)
            unique_radicals.update(per_char.get("radicals", []))

    if not char_metadata:
        return record

    annotated = dict(record)
    annotated[output_key] = {"characters": char_metadata}
    if deduplicate_radicals and unique_radicals:
        annotated[output_key]["unique_radicals"] = sorted(unique_radicals)
    return annotated


def main() -> None:
    args = parse_args()
    input_path = args.input_json
    output_path = (
        args.output_json
        if args.output_json is not None
        else input_path.with_name(f"{input_path.stem}_annotated{input_path.suffix}")
    )

    records = load_records(input_path, args.jsonl)

    dictionary = HanziDictionary()
    decomposer = HanziDecomposer()

    annotated_records = [
        annotate_record(
            record,
            dictionary,
            decomposer,
            args.text_key,
            args.output_key,
            args.include_pinyin,
            args.include_radicals,
            args.include_frequency,
            args.include_decomposition,
            args.deduplicate_radicals,
        )
        for record in records
    ]

    save_records(annotated_records, output_path, args.jsonl)
    LOGGER.info("Saved annotated dataset to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
