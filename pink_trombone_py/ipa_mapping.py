import subprocess
from typing import List
from .trombone import PinkTrombone
from .noise import RandomNoise
import numpy as np
import soundfile as sf

IPA_MAP = {
    'a': {'tongue_index': 20, 'tongue_diameter': 3.0, 'tenseness':0.6},
    'e': {'tongue_index': 15, 'tongue_diameter': 2.5, 'tenseness':0.6},
    'i': {'tongue_index': 12, 'tongue_diameter': 2.0, 'tenseness':0.6},
    'o': {'tongue_index': 22, 'tongue_diameter': 3.2, 'tenseness':0.6},
    'u': {'tongue_index': 24, 'tongue_diameter': 3.4, 'tenseness':0.6},
}


def espeak_to_ipa(text: str) -> str:
    result = subprocess.run(['espeak-ng','-q','--ipa=3', text], capture_output=True, text=True)
    return result.stdout.strip()


def synthesize_ipa(text: str, sample_rate: int = 48000) -> np.ndarray:
    ipa = espeak_to_ipa(text)
    rng = RandomNoise()
    trombone = PinkTrombone(sample_rate, rng, seed=42)
    duration_per_symbol = 0.3
    audio = []
    for ch in ipa:
        if ch not in IPA_MAP:
            continue
        params = IPA_MAP[ch]
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
