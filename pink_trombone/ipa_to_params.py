#!/usr/bin/env python3
"""Convert text to Pink Trombone parameters via espeak-ng IPA output.

This script is a small prototype demonstrating how IPA phonemes can be
translated into numeric values that control the Pink Trombone synthesizer.
The mapping below is intentionally small and should be extended to cover
all phonemes of interest.
"""

import argparse
import json
import subprocess
from typing import List, Dict

# Example mapping from IPA symbols to Pink Trombone parameter values.
# Values are chosen for demonstration and do not represent a finalized
# articulatory model.
PHONEME_MAP: Dict[str, Dict[str, float]] = {
    "a": {"tongue_index": 20.0, "tongue_diameter": 2.5, "velum_open": False},
    "i": {"tongue_index": 32.0, "tongue_diameter": 1.5, "velum_open": False},
    "u": {"tongue_index": 10.0, "tongue_diameter": 2.0, "velum_open": False},
    "e": {"tongue_index": 27.0, "tongue_diameter": 2.0, "velum_open": False},
    "o": {"tongue_index": 15.0, "tongue_diameter": 2.4, "velum_open": False},
    "ə": {"tongue_index": 22.0, "tongue_diameter": 2.3, "velum_open": False},
    "ʊ": {"tongue_index": 12.0, "tongue_diameter": 2.3, "velum_open": False},
    "m": {"tongue_index": 12.0, "tongue_diameter": 2.0, "velum_open": True},
    "l": {"tongue_index": 30.0, "tongue_diameter": 1.8, "velum_open": False},
    "h": {"tongue_index": 18.0, "tongue_diameter": 2.2, "velum_open": False},
}


def text_to_ipa(text: str) -> str:
    """Return espeak-ng IPA transcription for ``text``."""
    result = subprocess.run(
        ["espeak-ng", "--ipa", "-q", text], capture_output=True, text=True
    )
    return result.stdout.strip()


def ipa_to_params(ipa: str) -> List[Dict[str, float]]:
    """Convert an IPA string into a list of parameter dictionaries."""
    params = []
    for symbol in ipa.replace(" ", ""):
        mapping = PHONEME_MAP.get(symbol)
        if mapping:
            params.append(mapping)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert text to Pink Trombone parameters")
    parser.add_argument("text", help="Input text")
    args = parser.parse_args()

    ipa = text_to_ipa(args.text)
    frames = ipa_to_params(ipa)
    print(f"IPA: {ipa}")
    print(json.dumps(frames, indent=2))


if __name__ == "__main__":
    main()
