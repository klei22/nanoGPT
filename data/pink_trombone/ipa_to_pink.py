#!/usr/bin/env python3
"""Convert text to Pink Trombone parameter sequences using espeak."""
import json
import subprocess
import sys

# Example mapping from IPA symbols to Pink Trombone parameters.
# Extend this dictionary for better coverage.
PHONEME_TO_PARAMS = {
    "a": {"tongue_index": 16.0, "tongue_diameter": 2.8, "target_frequency": 140},
    "i": {"tongue_index": 20.0, "tongue_diameter": 1.6, "target_frequency": 220},
    "u": {"tongue_index": 12.0, "tongue_diameter": 2.4, "target_frequency": 110},
}


def text_to_ipa(text: str) -> str:
    """Call espeak to obtain an IPA transcription."""
    result = subprocess.run(
        ["espeak", "--ipa", "-q", text], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def ipa_to_params(ipa: str):
    """Convert an IPA string to a list of parameter dictionaries."""
    params = []
    for ch in ipa.replace(" ", ""):
        entry = PHONEME_TO_PARAMS.get(ch)
        if entry:
            params.append(entry)
    return params


def main():
    if len(sys.argv) < 2:
        print("Usage: ipa_to_pink.py 'some text'", file=sys.stderr)
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    ipa = text_to_ipa(text)
    param_list = ipa_to_params(ipa)
    json.dump(param_list, sys.stdout, indent=2)


if __name__ == "__main__":
    main()

