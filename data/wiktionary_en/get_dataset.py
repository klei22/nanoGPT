import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from tqdm import tqdm

DEFAULT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
DEFAULT_COMPRESSED_NAME = "raw-wiktextract-data.jsonl.gz"

def download_file(url: str, destination: Path) -> None:
    """Stream a remote file to disk with a progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"{destination} already exists, skipping download.")
        return

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024
        progress = tqdm(total=total_size or None, unit="iB", unit_scale=True, desc="Downloading dump")
        with open(destination, "wb") as fout:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fout.write(chunk)
                progress.update(len(chunk))
        progress.close()


def open_maybe_gzip(path: Path):
    """Open a text file that may be gzip-compressed."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_language_entries(dataset_path: Path, language: str) -> Iterable[dict]:
    """Yield entries whose `lang` field matches the requested language."""
    with open_maybe_gzip(dataset_path) as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("lang") == language:
                yield entry


def collect_examples(sense: dict) -> List[str]:
    examples = []
    for example in sense.get("examples", []):
        text = example.get("text")
        if text:
            examples.append(text)
    return examples


def format_sense(entry: dict, sense: dict) -> Optional[str]:
    glosses = sense.get("glosses") or sense.get("raw_glosses")
    examples = collect_examples(sense)

    if not glosses and not examples:
        return None

    parts: List[str] = []
    if entry.get("word"):
        parts.append(f"word: {entry['word']}")
    if entry.get("pos"):
        parts.append(f"part_of_speech: {entry['pos']}")
    if sense.get("tags"):
        parts.append(f"tags: {', '.join(sense['tags'])}")
    if glosses:
        parts.append("definition: " + " | ".join(glosses))
    if examples:
        parts.append("examples: " + " | ".join(examples))
    return "\n".join(parts)


def build_input_file(dataset_path: Path, output_path: Path, language: str, max_entries: Optional[int]) -> int:
    """Transform Wiktextract entries into the newline-delimited training file."""
    written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        progress = tqdm(total=max_entries or None, desc="Writing senses")
        for entry in iter_language_entries(dataset_path, language):
            for sense in entry.get("senses", []):
                block = format_sense(entry, sense)
                if not block:
                    continue
                if written:
                    fout.write("\n\n")
                fout.write(block)
                written += 1
                progress.update(1)
                if max_entries is not None and written >= max_entries:
                    progress.close()
                    fout.write("\n")
                    return written
            if max_entries is not None and written >= max_entries:
                break
        progress.close()
        if written:
            fout.write("\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and format English Wiktionary data.")
    parser.add_argument(
        "--download_url",
        type=str,
        default=DEFAULT_URL,
        help="URL for the Wiktextract JSONL dump to download (defaults to the full English dump).",
    )
    parser.add_argument(
        "--compressed_path",
        type=Path,
        default=Path(DEFAULT_COMPRESSED_NAME),
        help="Where to store the downloaded dump.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("input.txt"),
        help="Where to write the formatted text.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Filter entries by this language label from the dump.",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Optionally stop after writing this many senses.",
    )

    args = parser.parse_args()

    download_file(args.download_url, args.compressed_path)
    written = build_input_file(args.compressed_path, args.output_file, args.language, args.max_entries)
    print(f"Wrote {written} formatted senses to {args.output_file}")


if __name__ == "__main__":
    main()
