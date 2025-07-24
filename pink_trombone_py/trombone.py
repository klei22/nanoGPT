import numpy as np
from dataclasses import dataclass
from typing import List
from .glottis import Glottis
from .tract import Tract
from .tract_shaper import TractShaper
from .turbulence import TurbulencePoint
from .noise import NoiseSource

MAX_BLOCK_LEN = 512

@dataclass
class PinkTrombone:
    sample_rate: int
    rng: NoiseSource
    seed: int

    def __post_init__(self):
        glottis = Glottis(self.sample_rate, self.rng, self.seed)
        tract = Tract(glottis, 2*self.sample_rate, self.rng)
        self.shaper = TractShaper(tract)

    def synthesize(self, n_samples: int) -> np.ndarray:
        buf = np.zeros(n_samples, dtype=np.float32)
        p = 0
        while p < n_samples:
            block_len = min(n_samples - p, MAX_BLOCK_LEN)
            self.synthesize_block(buf[p:p+block_len])
            p += block_len
        return buf

    def synthesize_block(self, buf: np.ndarray):
        delta_time = len(buf)/self.sample_rate
        self.calculate_new_block_parameters(delta_time)
        for i in range(len(buf)):
            lambda1 = i/len(buf)
            lambda2 = (i+0.5)/len(buf)
            glottal_output = self.shaper.tract.glottis.step(lambda1)
            vocal1 = self.shaper.tract.step(glottal_output, lambda1)
            vocal2 = self.shaper.tract.step(glottal_output, lambda2)
            buf[i] = (vocal1 + vocal2) * 0.125

    def calculate_new_block_parameters(self, delta_time: float):
        self.shaper.tract.glottis.adjust_parameters(delta_time)
        self.shaper.adjust_tract_shape(delta_time)
        self.shaper.tract.calculate_new_block_parameters()

    # convenience wrappers
    def set_musical_note(self, semitone: float):
        self.shaper.tract.glottis.set_musical_note(semitone)

    def turbulence_points(self) -> List[TurbulencePoint]:
        return self.shaper.tract.turbulence_points
