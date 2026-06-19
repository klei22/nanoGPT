import argparse
import json
import os
import re
from pathlib import Path

from hanzipy.decomposer import HanziDecomposer
from hanzipy.dictionary import HanziDictionary

HANZI_REGEX = re.compile(r"[\u4e00-\u9fff]")


def is_hanzi_character(character):
    return bool(HANZI_REGEX.match(character))


def extract_unique_hanzi(text):
    seen = set()
    ordered = []
    for character in text:
        if character in seen:
            continue
        if not is_hanzi_character(character):
            continue
        seen.add(character)
        ordered.append(character)
    return ordered


def build_hanzi_annotation(character, dictionary, decomposer, cache):
    if character in cache:
        return cache[character]

    annotation = {
        "definitions": dictionary.definition_lookup(character),
        "pinyin": dictionary.get_pinyin(character),
        "frequency": dictionary.get_character_frequency(character),
        "decomposition": decomposer.decompose(character),
        "phonetic_regularity": dictionary.determine_phonetic_regularity(character),
    }

    radical_meanings = {}
    radicals = annotation["decomposition"].get("radical", [])
    for radical in radicals:
        if radical == "No glyph available":
            continue
        meaning = decomposer.get_radical_meaning(radical)
        if meaning:
            radical_meanings[radical] = meaning

    if radical_meanings:
        annotation["radical_meanings"] = radical_meanings

    cache[character] = annotation
    return annotation


def annotate_records(records, zh_key, dictionary, decomposer, cache):
    for record in records:
        translation = record.get("translation")
        if isinstance(translation, dict):
            zh_text = translation.get(zh_key)
        else:
            zh_text = record.get(zh_key)

        if not zh_text:
            record["annotated_hanzi"] = {
                "characters": [],
                "by_character": {},
            }
            continue

        characters = extract_unique_hanzi(zh_text)
        by_character = {
            character: build_hanzi_annotation(character, dictionary, decomposer, cache)
            for character in characters
        }

        record["annotated_hanzi"] = {
            "characters": characters,
            "by_character": by_character,
        }

    return records


def resolve_inputs(input_json, input_dir):
    if input_json and input_dir:
        raise ValueError("Provide only one of --input_json or --input_dir.")
    if not input_json and not input_dir:
        raise ValueError("Provide either --input_json or --input_dir.")

    if input_json:
        return [Path(input_json)]

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(input_path.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Annotate OPUS-100 Mandarin entries with Hanzi metadata using hanzipy.",
    )
    parser.add_argument("--input_json", type=str, help="Path to a single JSON file to annotate.")
    parser.add_argument("--input_dir", type=str, help="Directory of JSON files to annotate.")
    parser.add_argument(
        "--output_json",
        type=str,
        help="Output JSON path when annotating a single file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="annotated_json_output",
        help="Output directory for annotated JSON files when using --input_dir.",
    )
    parser.add_argument(
        "--zh_key",
        type=str,
        default="zh",
        help="Language key for Mandarin Chinese in the translation field (default: zh).",
    )

    args = parser.parse_args()

    input_files = resolve_inputs(args.input_json, args.input_dir)

    if args.input_json and not args.output_json:
        raise ValueError("--output_json is required when using --input_json.")

    output_dir = Path(args.output_dir)
    if args.input_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = HanziDictionary()
    decomposer = HanziDecomposer()
    cache = {}

    for input_path in input_files:
        with input_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        annotated = annotate_records(records, args.zh_key, dictionary, decomposer, cache)

        if args.input_json:
            output_path = Path(args.output_json)
        else:
            output_path = output_dir / input_path.name

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(annotated, handle, ensure_ascii=False, indent=2)

        print(f"Annotated {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
