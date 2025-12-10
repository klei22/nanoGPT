import argparse
import gzip
import json
import os
from typing import Iterable, List, Optional

import requests
from tqdm import tqdm


DEFAULT_DATA_URL = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl"
DEFAULT_DATASET_PATH = "kaikki.org-dictionary-English.jsonl"
DEFAULT_OUTPUT_FILE = "input.txt"


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def extract_terms(items: Iterable) -> List[str]:
    terms: List[str] = []
    for item in items:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, dict):
            candidate = item.get("word") or item.get("text") or item.get("sense")
        else:
            candidate = None

        if candidate:
            terms.append(clean_text(str(candidate)))
    return terms


def download_dataset(url: str, destination: str) -> None:
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping download")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)
    progress_bar.close()
    print(f"Downloaded dataset to {destination}")


def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def format_entry(entry: dict) -> Optional[str]:
    if entry.get("lang_code") != "en":
        return None

    word = entry.get("word")
    if not word:
        return None

    senses = entry.get("senses", [])
    formatted_senses: List[str] = []
    for sense in senses:
        parts: List[str] = []
        glosses = sense.get("glosses") or sense.get("raw_glosses")
        if glosses:
            parts.append("; ".join(clean_text(gloss) for gloss in glosses if gloss))

        examples = [
            clean_text(example.get("text"))
            for example in sense.get("examples", [])
            if isinstance(example, dict) and example.get("text")
        ]
        if examples:
            parts.append("Examples: " + " | ".join(examples))

        synonyms = extract_terms(sense.get("synonyms", []))
        if synonyms:
            parts.append("Synonyms: " + ", ".join(synonyms))

        if parts:
            formatted_senses.append("- " + " ".join(parts))

    if not formatted_senses:
        return None

    pos = entry.get("pos")
    header = clean_text(word)
    if pos:
        header += f" ({pos})"

    etymology = entry.get("etymology_text")
    lines = [header]
    if etymology:
        lines.append("Etymology: " + clean_text(etymology))

    lines.extend(formatted_senses)
    return "\n".join(lines)


def process_dataset(dataset_path: str, output_path: str, max_entries: Optional[int] = None) -> None:
    written = 0
    with open_maybe_gzip(dataset_path) as dataset_file, open(output_path, "w", encoding="utf-8") as output_file:
        for line in dataset_file:
            entry = json.loads(line)
            formatted = format_entry(entry)
            if formatted:
                output_file.write(formatted)
                output_file.write("\n\n")
                written += 1
            if max_entries is not None and written >= max_entries:
                break
    print(f"Wrote {written} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the English Wiktionary dump and convert it to input.txt"
    )
    parser.add_argument(
        "--data_url",
        type=str,
        default=DEFAULT_DATA_URL,
        help="URL pointing to the wiktionary JSONL export",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Where to store the downloaded JSONL file",
    )
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the generated input.txt file",
    )
    parser.add_argument(
        "-n",
        "--max_entries",
        type=int,
        default=None,
        help="Optional limit useful for debugging",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Process an existing dataset file without downloading",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset(args.data_url, args.dataset_path)

    process_dataset(args.dataset_path, args.output_text_file, args.max_entries)


if __name__ == "__main__":
    main()
