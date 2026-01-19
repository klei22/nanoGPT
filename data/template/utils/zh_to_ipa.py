#!/usr/bin/env python3
# zh2ipa.py

import subprocess  # kept (even though unused) to match your original imports
from dragonmapper import hanzi
import jieba
import argparse
import re
import json


WRAP_PREFIX = "[[[[["
WRAP_SUFFIX = "]]]]]"


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def is_chinese_token(token: str) -> bool:
    # Keeps your original behavior: "Chinese" means "contains any simplified Hanzi"
    # (This will miss pure-traditional-only text; expand if you want.)
    return any(hanzi.is_simplified(ch) for ch in token)


def transcribe_chinese(sentence: str) -> str:
    """Transcribe a Chinese sentence into its phonemes using dragonmapper."""
    try:
        result = hanzi.to_ipa(sentence)
        return "".join(result)
    except Exception as e:
        return f"Error in transcribing Chinese: {str(e)}"


def handle_mixed_language(word: str, wrapper: bool = True) -> str:
    """Handle a word with potential Chinese, other language, or number content."""
    if word.isdigit():  # numbers: passthrough
        return word
    elif is_chinese_token(word):  # Chinese: IPA
        return transcribe_chinese(word)
    else:  # Non-Chinese: wrap or passthrough
        return f"{WRAP_PREFIX}{word}{WRAP_SUFFIX}" if wrapper else word


def transcribe_multilingual(
    data,
    output_file: str,
    json_inplace_update: bool = False,
    json_input_field: str = "sentence",
    json_output_field: str = "sentence_ipa",
    wrapper: bool = True,
):
    """
    Transcribe multilingual sentences (Chinese + non-Chinese passthrough/wrap) and save to a file.

    Also computes byte counts:
      - transcribed_bytes: UTF-8 bytes of ORIGINAL tokens that were transcribed (Chinese tokens)
      - not_transcribed_bytes: UTF-8 bytes of ORIGINAL tokens that were not transcribed
        (includes Latin words, digits, punctuation, etc.)
    These counts are based on ORIGINAL text tokens, so wrapper overhead is automatically excluded.

    Returns:
        stats dict with transcribed_bytes, not_transcribed_bytes, total_bytes, and percents.
    """
    transcribed_bytes = 0
    not_transcribed_bytes = 0

    def process_sentence(sentence: str) -> str:
        nonlocal transcribed_bytes, not_transcribed_bytes

        # Split sentence using jieba (your original behavior)
        seg_list = jieba.cut(sentence, cut_all=False)
        seg_sentence = "".join(seg_list)

        # Split but keep punctuation
        words = re.findall(r"\w+|[^\w\s]", seg_sentence, re.UNICODE)

        out_parts = []
        for tok in words:
            tok_bytes = utf8_len(tok)

            if re.match(r"\w+", tok):
                # word-ish token
                if tok.isdigit():
                    not_transcribed_bytes += tok_bytes
                elif is_chinese_token(tok):
                    transcribed_bytes += tok_bytes
                else:
                    not_transcribed_bytes += tok_bytes

                out_parts.append(handle_mixed_language(tok, wrapper=wrapper))
            else:
                # punctuation / symbols
                not_transcribed_bytes += tok_bytes
                out_parts.append(tok)

        return " ".join(out_parts)

    if json_inplace_update:
        # In-place update for JSON data
        for item in data:
            if json_input_field in item:
                sentence = item[json_input_field]
                item[json_output_field] = process_sentence(sentence)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"In-place JSON transcription saved to {output_file}")

    else:
        # Standard transcription to plain text output (one line per item)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                if isinstance(item, dict):
                    sentence = item.get(json_input_field, "")
                else:
                    sentence = item

                transcription_result = process_sentence(sentence)
                f.write(transcription_result + "\n")
                print(transcription_result)

    total_bytes = transcribed_bytes + not_transcribed_bytes
    pct_transcribed = (transcribed_bytes / total_bytes * 100.0) if total_bytes else 0.0
    pct_not = (not_transcribed_bytes / total_bytes * 100.0) if total_bytes else 0.0

    stats = {
        "transcribed_bytes": transcribed_bytes,
        "not_transcribed_bytes": not_transcribed_bytes,
        "total_bytes": total_bytes,
        "pct_transcribed": pct_transcribed,
        "pct_not_transcribed": pct_not,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe multilingual sentences into IPA phonemes (Chinese via dragonmapper), with byte coverage stats."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file containing sentences in json or text format."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file for IPA transcription."
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["json", "text"],
        default="json",
        help='Type of input file: "json" or "text" (default: json)'
    )
    parser.add_argument(
        "-j", "--json_inplace_update",
        action="store_true",
        help="Process JSON input and add IPA to the same JSON entries"
    )
    parser.add_argument(
        "--json_input_field",
        default="sentence",
        help="JSON field to read from (default: sentence)"
    )
    parser.add_argument(
        "--json_output_field",
        default="sentence_ipa",
        help="JSON field to write IPA to (default: sentence_ipa)"
    )
    parser.add_argument(
        "--wrapper",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Wrap non-Chinese tokens as [[[[[...]]]]] (default: true). Use --no-wrapper to leave them unchanged."
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Optional: write stats as JSON to this path (in addition to printing)."
    )

    args = parser.parse_args()

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            if args.input_type == "json":
                data = json.load(f)
            else:
                data = [line.rstrip("\n") for line in f.readlines()]

        stats = transcribe_multilingual(
            data=data,
            output_file=args.output_file,
            json_inplace_update=args.json_inplace_update,
            json_input_field=args.json_input_field,
            json_output_field=args.json_output_field,
            wrapper=args.wrapper,
        )

        # Print summary stats (wrapper overhead is automatically excluded because we count ORIGINAL token bytes)
        print("\n=== Byte Coverage Stats (based on ORIGINAL tokens) ===")
        print(f"Transcribed bytes      : {stats['transcribed_bytes']}")
        print(f"Not transcribed bytes  : {stats['not_transcribed_bytes']}")
        print(f"Total bytes (counted)  : {stats['total_bytes']}")
        print(f"% transcribed          : {stats['pct_transcribed']:.2f}%")
        print(f"% not transcribed      : {stats['pct_not_transcribed']:.2f}%")

        if args.stats_json:
            with open(args.stats_json, "w", encoding="utf-8") as sf:
                json.dump(stats, sf, ensure_ascii=False, indent=2)
            print(f"Stats JSON written to: {args.stats_json}")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.input_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

