#!/usr/bin/env python3
import subprocess
from konlpy.tag import Okt
import argparse
import re
import json


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
                return "[[[[[" + sentence + "]]]]]"
            return sentence

        return transcription

    except Exception as e:
        # Keep behavior consistent: return an error string
        return f"Error in transcribing Korean: {str(e)}"


def handle_mixed_language(word, wrapper=False):
    """Handle a word with potential Korean, other language, or number content."""
    if word.isdigit():  # Detect numbers (pass through unchanged)
        return word
    elif any('가' <= char <= '힣' for char in word):  # Detect Korean
        return transcribe_korean(word, wrapper=wrapper)
    else:  # Non-Korean word
        if wrapper:
            return "[[[[[" + word + "]]]]]"
        return word


def transcribe_plain_text(text, wrapper=False):
    """Transcribe a plain text string into IPA, leaving non-Korean as-is (or wrapped)."""
    result = []
    words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    for word in words:
        if re.match(r'\w+', word):
            result.append(handle_mixed_language(word, wrapper=wrapper))
        else:
            result.append(word)
    return " ".join(result)


def transcribe_multilingual(sentences, input_json_key=None, output_json_key='ipa', wrapper=False):
    """
    Transcribe multilingual sentences and update JSON data directly.

    Args:
        sentences: JSON string or a loaded JSON object.
        input_json_key: Key to extract sentences from in a JSON.
        output_json_key: Key to store IPA transcription in the JSON (default: 'ipa').

    Returns:
        The modified JSON string with IPA transcriptions added.
    """
    try:
        data = json.loads(sentences) if isinstance(sentences, str) else sentences
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")

        for item in data:
            if input_json_key in item:
                sentence = item[input_json_key]
                transcription_result = transcribe_plain_text(sentence, wrapper=wrapper)
                item[output_json_key] = transcription_result  # Update directly
                print(transcription_result)
            else:
                print(f"Warning: Key '{input_json_key}' not found in item: {item}")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return None

    return json.dumps(data, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe multilingual text or JSON into IPA phonemes (Korean via espeak-ng).'
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
        help="Wrap unparseable text with [[[[[square brackets]]]]], for later recovery."
    )

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_content = f.read()

        # ---- TEXT MODE ----
        if args.text_input:
            transcription = transcribe_plain_text(
                input_content,
                wrapper=args.wrapper
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
                wrapper=args.wrapper
            )

            if updated_json_data:
                # Default behavior: overwrite original JSON
                if args.text_output:
                    with open(args.text_output, 'w', encoding='utf-8') as f:
                        f.write(updated_json_data)
                    print(f"Wrote JSON output to '{args.text_output}'")
                else:
                    with open(args.input_file, 'w', encoding='utf-8') as f:
                        f.write(updated_json_data)
                    print(f"Successfully updated JSON data in '{args.input_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

