import subprocess
from typing import List
from .trombone import PinkTrombone
from .noise import RandomNoise
import numpy as np
import soundfile as sf

# Mapping from a small set of canonical vowels to tract parameters.
IPA_MAP = {
    'a': {'tongue_index': 20, 'tongue_diameter': 3.0, 'tenseness': 0.6},
    'e': {'tongue_index': 15, 'tongue_diameter': 2.5, 'tenseness': 0.6},
    'i': {'tongue_index': 12, 'tongue_diameter': 2.0, 'tenseness': 0.6},
    'o': {'tongue_index': 22, 'tongue_diameter': 3.2, 'tenseness': 0.6},
    'u': {'tongue_index': 24, 'tongue_diameter': 3.4, 'tenseness': 0.6},
}

# Normalize various IPA vowel symbols to a small canonical set.  This keeps the
# synthesis code simple while allowing multi-character sequences such as ``aɪ``
# or ``oʊ`` to be interpreted as consecutive vowels.
_VOWEL_EQUIV = {
    "a": "a", "ɑ": "a", "æ": "a", "ʌ": "a", "ɐ": "a", "ɜ": "a", "ə": "a",
    "e": "e", "ɛ": "e", "ɚ": "e",
    "i": "i", "ɪ": "i", "ɨ": "i",
    "o": "o", "ɔ": "o",
    "u": "u", "ʊ": "u",
}

_STRESS_MARKS = {"ˈ", "ˌ"}
_ZERO_WIDTH_JOINER = "\u200d"


def espeak_to_ipa(text: str) -> str:
    result = subprocess.run(['espeak-ng','-q','--ipa=3', text], capture_output=True, text=True)
    return result.stdout.strip()


def _extract_canonical_vowels(ipa: str) -> List[str]:
    """Return a list of canonical vowel symbols from an IPA string."""
    vowels: List[str] = []
    for ch in ipa:
        if ch in _STRESS_MARKS or ch == _ZERO_WIDTH_JOINER or ch.isspace():
            continue
        canonical = _VOWEL_EQUIV.get(ch)
        if canonical:
            vowels.append(canonical)
    return vowels


def synthesize_ipa(text: str, sample_rate: int = 48000) -> np.ndarray:
    ipa = espeak_to_ipa(text)
    vowels = _extract_canonical_vowels(ipa)
    rng = RandomNoise()
    trombone = PinkTrombone(sample_rate, rng, seed=42)
    duration_per_symbol = 0.3
    audio = []
    for ch in vowels:
        params = IPA_MAP.get(ch)
        if not params:
            continue
        trombone.shaper.tongue_index = params['tongue_index']
        trombone.shaper.tongue_diameter = params['tongue_diameter']
        trombone.shaper.tract.glottis.target_tenseness = params['tenseness']
        samples = trombone.synthesize(int(sample_rate*duration_per_symbol))
        audio.append(samples)
    if audio:
        return np.concatenate(audio)
    else:
        return np.zeros(0)


def synthesize_to_wav(text: str, path: str, sample_rate: int = 48000):
    audio = synthesize_ipa(text, sample_rate)
    sf.write(path, audio, sample_rate)
