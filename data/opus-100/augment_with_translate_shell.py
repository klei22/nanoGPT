import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


class TranslateShellError(RuntimeError):
    """Raised when translate-shell is unavailable or returns an error."""


def translate_word(word: str, from_lang: str, to_lang: str, timeout: float) -> str:
    """Translate a single word using translate-shell.

    Args:
        word: The word to translate.
        from_lang: Source language code (e.g. ``en``).
        to_lang: Target language code (e.g. ``es``).
        timeout: Seconds to wait for translate-shell to respond.

    Returns:
        The translated word, or an empty string if translation fails.

    Raises:
        TranslateShellError: If translate-shell is not installed or exits with an error.
    """

    try:
        completed = subprocess.run(
            ["trans", "-b", f"{from_lang}:{to_lang}", word],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TranslateShellError(
            "translate-shell (trans) is required. Install it before running this script."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TranslateShellError(
            f"translate-shell timed out translating '{word}' from {from_lang} to {to_lang}."
        ) from exc

    if completed.returncode != 0:
        raise TranslateShellError(
            f"translate-shell returned error code {completed.returncode}: {completed.stderr.strip()}"
        )

    return completed.stdout.strip()


def build_word_translations(
    source_sentence: str, from_lang: str, to_lang: str, timeout: float
) -> List[Dict[str, str]]:
    """Translate each whitespace-delimited token in a sentence.

    Args:
        source_sentence: The sentence to translate word-by-word.
        from_lang: Source language code.
        to_lang: Target language code.
        timeout: Seconds to wait for each translate-shell call.

    Returns:
        A list of dictionaries with ``source`` and ``translation`` keys.
    """

    translations: List[Dict[str, str]] = []
    for word in source_sentence.split():
        translated_word = translate_word(word, from_lang, to_lang, timeout)
        translations.append({"source": word, "translation": translated_word})
    return translations


def augment_entries(
    entries: List[Dict[str, Any]],
    from_lang: str,
    to_lang: str,
    timeout: float,
    max_items: Optional[int],
) -> List[Dict[str, Any]]:
    """Augment OPUS-100 translation entries with translate-shell outputs."""

    augmented: List[Dict[str, Any]] = []
    iterable = entries if max_items is None else entries[:max_items]

    for item in tqdm(iterable, desc="Augmenting entries"):
        translation = item.get("translation", {})
        source_sentence = translation.get(from_lang)
        target_sentence = translation.get(to_lang)

        if source_sentence is None or target_sentence is None:
            # Skip malformed entries but keep original payload for visibility.
            augmented.append(item)
            continue

        word_translations = build_word_translations(
            source_sentence, from_lang, to_lang, timeout
        )

        augmented_item = {
            **item,
            "translation": translation,
            "translate_shell": {
                "from_lang": from_lang,
                "to_lang": to_lang,
                "word_translations": word_translations,
            },
        }
        augmented.append(augmented_item)

    return augmented


def emit_text_dataset(
    entries: List[Dict[str, Any]],
    output_text_file: Path,
    from_lang: str,
    to_lang: str,
) -> None:
    """Emit a text dataset with source, word-level translations, and target."""

    with output_text_file.open("w", encoding="utf-8") as output_file:
        for item in entries:
            translation = item.get("translation", {})
            word_translations = item.get("translate_shell", {}).get("word_translations", [])
            source_sentence = translation.get(from_lang, "").replace("\n", " ").strip()
            target_sentence = translation.get(to_lang, "").replace("\n", " ").strip()

            translated_words = " ".join(
                f"{entry['source']}:{entry['translation']}" for entry in word_translations
            )

            output_file.write(source_sentence)
            output_file.write("\n")
            output_file.write(translated_words)
            output_file.write("\n")
            output_file.write(target_sentence)
            output_file.write("\n\n")


def load_json_entries(input_json: Path) -> List[Dict[str, Any]]:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def save_json(entries: List[Dict[str, Any]], output_json: Path) -> None:
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Augment OPUS-100 JSON entries with translate-shell word-level translations "
            "and emit a text dataset containing the source sentence, per-word translations, "
            "and the target sentence."
        )
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to the OPUS-100 JSON file produced by get_dataset.py",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("augmented_translate_shell.json"),
        help="Path to the augmented JSON output file.",
    )
    parser.add_argument(
        "--output_text_file",
        type=Path,
        default=Path("augmented_translate_shell.txt"),
        help=(
            "Path to the emitted text dataset containing the source sentence, each word's "
            "translation, and the target sentence."
        ),
    )
    parser.add_argument(
        "-f",
        "--from_lang_code",
        type=str,
        required=True,
        help="Source language code (e.g. en)",
    )
    parser.add_argument(
        "-t",
        "--to_lang_code",
        type=str,
        required=True,
        help="Target language code (e.g. es)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Optionally limit the number of entries to augment for quicker experiments.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each translate-shell call before failing.",
    )

    args = parser.parse_args()

    entries = load_json_entries(args.input_json)
    augmented_entries = augment_entries(
        entries,
        args.from_lang_code,
        args.to_lang_code,
        args.timeout,
        args.max_items,
    )

    save_json(augmented_entries, args.output_json)
    emit_text_dataset(augmented_entries, args.output_text_file, args.from_lang_code, args.to_lang_code)


if __name__ == "__main__":
    main()
