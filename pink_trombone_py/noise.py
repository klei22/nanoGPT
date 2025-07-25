import numpy as np

class NoiseSource:
    def noise(self) -> float:
        raise NotImplementedError

class RandomNoise(NoiseSource):
    def noise(self) -> float:
        return np.random.rand()

class LoopedNoiseBuffer(NoiseSource):
    def __init__(self, loop_size: int, rng: NoiseSource):
        self.noise_buf = [2*rng.noise()-1 for _ in range(loop_size)]
        self.current = 0
    def noise(self) -> float:
        if self.current >= len(self.noise_buf):
            self.current = 0
        val = self.noise_buf[self.current]
        self.current += 1
        return val

class BiquadIirFilter:
    def __init__(self, b0,b1,b2,a0,a1,a2):
        self.nb0 = b0/a0
        self.nb1 = b1/a0
        self.nb2 = b2/a0
        self.na1 = a1/a0
        self.na2 = a2/a0
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0
    def filter(self, x: float) -> float:
        y = self.nb0*x + self.nb1*self.x1 + self.nb2*self.x2 - self.na1*self.y1 - self.na2*self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y

def new_bandpass_filter(f0, q, sample_rate):
    w0 = 2*np.pi*f0/sample_rate
    alpha = np.sin(w0)/(2*q)
    b0,b1,b2 = alpha,0,-alpha
    a0 = 1+alpha
    a1 = -2*np.cos(w0)
    a2 = 1-alpha
    return BiquadIirFilter(b0,b1,b2,a0,a1,a2)

def new_filtered_noise_source(f0,q,sample_rate,loop_size,rng):
    white_noise = LoopedNoiseBuffer(loop_size,rng)
    filt = new_bandpass_filter(f0,q,sample_rate)
    def src():
        return filt.filter(white_noise.noise())
    return src
