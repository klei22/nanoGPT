import subprocess
from typing import List
from .trombone import PinkTrombone
from .noise import RandomNoise
import numpy as np
import soundfile as sf

# Mapping from a small set of canonical symbols to tract parameters.  Each entry
# may specify tongue shape, lip closure, velum state and glottal tenseness.
IPA_MAP = {
    'a': {'tongue_index': 20, 'tongue_diameter': 3.0, 'tenseness': 0.6, 'lip_closure': 0.0, 'velum_open': False},
    'e': {'tongue_index': 15, 'tongue_diameter': 2.5, 'tenseness': 0.6, 'lip_closure': 0.0, 'velum_open': False},
    'i': {'tongue_index': 12, 'tongue_diameter': 2.0, 'tenseness': 0.6, 'lip_closure': 0.0, 'velum_open': False},
    'o': {'tongue_index': 22, 'tongue_diameter': 3.2, 'tenseness': 0.6, 'lip_closure': 0.0, 'velum_open': False},
    'u': {'tongue_index': 24, 'tongue_diameter': 3.4, 'tenseness': 0.6, 'lip_closure': 0.0, 'velum_open': False},
    # Bilabial voiced stop
    'b': {'lip_closure': 1.0, 'tenseness': 0.7, 'velum_open': False},
    # Bilabial nasal
    'm': {'lip_closure': 1.0, 'tenseness': 0.7, 'velum_open': True},
    # Labiodental fricative
    'f': {'lip_closure': 0.6, 'tenseness': 0.3, 'velum_open': False},
}

# Normalize various IPA vowel symbols to a small canonical set.  This keeps the
# synthesis code simple while allowing multi-character sequences such as ``aɪ``
# or ``oʊ`` to be interpreted as consecutive vowels.  Consonants ``b``, ``m``
# and ``f`` are passed through unchanged.
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


def _extract_canonical_symbols(ipa: str) -> List[str]:
    """Return a list of canonical phoneme symbols from an IPA string."""
    phones: List[str] = []
    for ch in ipa:
        if ch in _STRESS_MARKS or ch == _ZERO_WIDTH_JOINER or ch.isspace():
            continue
        canonical = _VOWEL_EQUIV.get(ch)
        if canonical:
            phones.append(canonical)
        elif ch in {'f', 'b', 'm'}:
            phones.append(ch)
    return phones


def synthesize_ipa(text: str, sample_rate: int = 48000) -> np.ndarray:
    ipa = espeak_to_ipa(text)
    symbols = _extract_canonical_symbols(ipa)
    rng = RandomNoise()
    trombone = PinkTrombone(sample_rate, rng, seed=42)
    duration_per_symbol = 0.3
    audio = []
    for ch in symbols:
        params = IPA_MAP.get(ch)
        if not params:
            continue
        if 'tongue_index' in params:
            trombone.shaper.tongue_index = params['tongue_index']
        if 'tongue_diameter' in params:
            trombone.shaper.tongue_diameter = params['tongue_diameter']
        if 'tenseness' in params:
            trombone.shaper.tract.glottis.target_tenseness = params['tenseness']
        if 'lip_closure' in params:
            trombone.shaper.set_lip_closure(params['lip_closure'])
        if 'velum_open' in params:
            trombone.shaper.set_velum_open(params['velum_open'])
        samples = trombone.synthesize(int(sample_rate*duration_per_symbol))
        audio.append(samples)
    if audio:
        return np.concatenate(audio)
    else:
        return np.zeros(0)


def synthesize_to_wav(text: str, path: str, sample_rate: int = 48000):
    audio = synthesize_ipa(text, sample_rate)
    sf.write(path, audio, sample_rate)
