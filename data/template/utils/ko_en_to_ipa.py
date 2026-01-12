#!/usr/bin/env python3
import subprocess
from konlpy.tag import Okt
import argparse
import re
import json

WRAP_PREFIX = "[[[[["
WRAP_SUFFIX = "]]]]]"


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def is_korean_token(token: str) -> bool:
    return any('가' <= ch <= '힣' for ch in token)


def transcribe_korean(sentence, wrapper=False):
    """Transcribe a Korean sentence into its phonemes using KoNLPy (Okt) + espeak-ng."""
    okt = Okt()
    tokens = okt.morphs(sentence)
    tokenized_sentence = ' '.join(tokens)

    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "ko", "--ipa", tokenized_sentence],
            capture_output=True,
            text=True
        )

        # Remove unwanted characters
        transcription = result.stdout.strip().replace("ㆍ", " ")

        # Check for failed transcription markers
        if "(en)" in transcription or "(ko)" in transcription:
            if wrapper:
                return f"{WRAP_PREFIX}{sentence}{WRAP_SUFFIX}"
            return sentence

        return transcription

    except Exception as e:
        # Keep behavior consistent: return an error string
        return f"Error in transcribing Korean: {str(e)}"


def handle_mixed_language(word, wrapper=False):
    """Handle a word with potential Korean, other language, or number content."""
    if word.isdigit():  # numbers pass through unchanged
        return word
    elif is_korean_token(word):
        return transcribe_korean(word, wrapper=wrapper)
    else:  # Non-Korean
        if wrapper:
            return f"{WRAP_PREFIX}{word}{WRAP_SUFFIX}"
        return word


def transcribe_plain_text(
    text,
    wrapper=False,
    stats=None,
):
    """
    Transcribe a plain text string into IPA, leaving non-Korean as-is (or wrapped).

    If stats dict is provided, it will be updated with:
      - transcribed_bytes: UTF-8 bytes of ORIGINAL tokens that were transcribed (Korean tokens only)
      - not_transcribed_bytes: UTF-8 bytes of ORIGINAL tokens not transcribed (includes Latin, digits, punctuation)
    Counts are based on ORIGINAL tokens, so wrapper overhead is excluded automatically.
    """
    if stats is None:
        stats = {}

    stats.setdefault("transcribed_bytes", 0)
    stats.setdefault("not_transcribed_bytes", 0)

    out = []
    words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    for tok in words:
        tok_bytes = utf8_len(tok)

        if re.match(r'\w+', tok):
            if tok.isdigit():
                stats["not_transcribed_bytes"] += tok_bytes
            elif is_korean_token(tok):
                stats["transcribed_bytes"] += tok_bytes
            else:
                stats["not_transcribed_bytes"] += tok_bytes

            out.append(handle_mixed_language(tok, wrapper=wrapper))
        else:
            # punctuation/symbols
            stats["not_transcribed_bytes"] += tok_bytes
            out.append(tok)

    return " ".join(out)


def transcribe_multilingual(sentences, input_json_key=None, output_json_key='ipa', wrapper=False, stats=None):
    """
    Transcribe multilingual sentences and update JSON data directly.

    Returns the modified JSON string with IPA transcriptions added.
    If stats dict is provided, it will be updated with byte coverage counts.
    """
    if stats is None:
        stats = {}
    stats.setdefault("transcribed_bytes", 0)
    stats.setdefault("not_transcribed_bytes", 0)

    try:
        data = json.loads(sentences) if isinstance(sentences, str) else sentences
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")

        for item in data:
            if input_json_key in item:
                sentence = item[input_json_key]
                transcription_result = transcribe_plain_text(sentence, wrapper=wrapper, stats=stats)
                item[output_json_key] = transcription_result
                print(transcription_result)
            else:
                print(f"Warning: Key '{input_json_key}' not found in item: {item}")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return None

    return json.dumps(data, ensure_ascii=False, indent=4)


def finalize_and_print_stats(stats, stats_json_path=None):
    transcribed = int(stats.get("transcribed_bytes", 0))
    not_tx = int(stats.get("not_transcribed_bytes", 0))
    total = transcribed + not_tx
    pct_tx = (transcribed / total * 100.0) if total else 0.0
    pct_not = (not_tx / total * 100.0) if total else 0.0

    out_stats = {
        "transcribed_bytes": transcribed,
        "not_transcribed_bytes": not_tx,
        "total_bytes": total,
        "pct_transcribed": pct_tx,
        "pct_not_transcribed": pct_not,
    }

    print("\n=== Byte Coverage Stats (based on ORIGINAL tokens) ===")
    print(f"Transcribed bytes      : {out_stats['transcribed_bytes']}")
    print(f"Not transcribed bytes  : {out_stats['not_transcribed_bytes']}")
    print(f"Total bytes (counted)  : {out_stats['total_bytes']}")
    print(f"% transcribed          : {out_stats['pct_transcribed']:.2f}%")
    print(f"% not transcribed      : {out_stats['pct_not_transcribed']:.2f}%")

    if stats_json_path:
        with open(stats_json_path, "w", encoding="utf-8") as sf:
            json.dump(out_stats, sf, ensure_ascii=False, indent=2)
        print(f"Stats JSON written to: {stats_json_path}")

    return out_stats


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe multilingual text or JSON into IPA phonemes (Korean via espeak-ng), with byte coverage stats.'
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input file (JSON or plain text).'
    )

    parser.add_argument(
        '--input_json_key',
        type=str,
        help='Key of the text field in JSON (required unless --text_input).'
    )

    parser.add_argument(
        '--output_json_key',
        type=str,
        default='ipa',
        help='Key to store the IPA transcription in the JSON output (default: "ipa").'
    )

    parser.add_argument(
        '--text_input',
        action='store_true',
        help='Treat input_file as plain text instead of JSON.'
    )

    parser.add_argument(
        '--text_output',
        type=str,
        help='Write output to a text file (or JSON file in JSON mode) instead of overwriting input.'
    )

    parser.add_argument(
        "--wrapper",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Wrap unparseable/non-target tokens with [[[[[...]]]]]. Use --no-wrapper to leave them unchanged."
    )

    parser.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Optional: write byte coverage stats as JSON to this path (in addition to printing)."
    )

    args = parser.parse_args()

    stats = {"transcribed_bytes": 0, "not_transcribed_bytes": 0}

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_content = f.read()

        # ---- TEXT MODE ----
        if args.text_input:
            transcription = transcribe_plain_text(
                input_content,
                wrapper=args.wrapper,
                stats=stats
            )

            if args.text_output:
                with open(args.text_output, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"Wrote transcription to '{args.text_output}'")
            else:
                print(transcription)

        # ---- JSON MODE (DEFAULT) ----
        else:
            if not args.input_json_key:
                raise ValueError("--input_json_key is required unless --text_input is used")

            updated_json_data = transcribe_multilingual(
                input_content,
                args.input_json_key,
                args.output_json_key,
                wrapper=args.wrapper,
                stats=stats
            )

            if updated_json_data:
                if args.text_output:
                    with open(args.text_output, 'w', encoding='utf-8') as f:
                        f.write(updated_json_data)
                    print(f"Wrote JSON output to '{args.text_output}'")
                else:
                    with open(args.input_file, 'w', encoding='utf-8') as f:
                        f.write(updated_json_data)
                    print(f"Successfully updated JSON data in '{args.input_file}'")

        finalize_and_print_stats(stats, stats_json_path=args.stats_json)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

