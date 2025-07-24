import numpy as np
from dataclasses import dataclass, field
from typing import List
from .glottis import Glottis
from .math_utils import interpolate, sqr, move_towards
from .noise import new_filtered_noise_source, NoiseSource
from .transient import Transient
from .turbulence import TurbulencePoint

N = 44
BLADE_START = 10
TIP_START = 32
LIP_START = 39
NOSE_LEN = 28
NOSE_START = N - NOSE_LEN + 1
GLOTTAL_REFLECTION = 0.75
LIP_REFLECTION = -0.85

@dataclass
class Tract:
    glottis: Glottis
    sample_rate: int
    rng: NoiseSource

    N = N
    BLADE_START = BLADE_START
    TIP_START = TIP_START
    LIP_START = LIP_START
    NOSE_START = NOSE_START
    NOSE_LEN = NOSE_LEN

    def __post_init__(self):
        self.frication_noise_source = new_filtered_noise_source(1000.0,0.5,self.sample_rate,0x8000,self.rng)
        self.sample_count = 0
        self.time = 0.0
        self.left = np.zeros(N)
        self.right = np.zeros(N)
        self.reflection = np.zeros(N)
        self.new_reflection = np.zeros(N)
        self.junction_output_right = np.zeros(N)
        self.junction_output_left = np.zeros(N+1)
        self.max_amplitude = np.zeros(N)
        self.diameter = np.zeros(N)
        self.transients: List[Transient] = []
        self.turbulence_points: List[TurbulencePoint] = []
        self.nose_right = np.zeros(NOSE_LEN)
        self.nose_left = np.zeros(NOSE_LEN)
        self.nose_junction_output_right = np.zeros(NOSE_LEN)
        self.nose_junction_output_left = np.zeros(NOSE_LEN+1)
        self.nose_reflection = np.zeros(NOSE_LEN)
        self.nose_diameter = np.zeros(NOSE_LEN)
        self.nose_max_amplitude = np.zeros(NOSE_LEN)
        self.reflection_left = 0.0
        self.reflection_right = 0.0
        self.new_reflection_left = 0.0
        self.new_reflection_right = 0.0
        self.reflection_nose = 0.0
        self.new_reflection_nose = 0.0

    def calculate_nose_reflections(self):
        a = np.maximum(1e-6, sqr(self.nose_diameter))
        for i in range(1, NOSE_LEN):
            self.nose_reflection[i] = (a[i-1]-a[i])/(a[i-1]+a[i])

    def calculate_new_block_parameters(self):
        self.calculate_main_tract_reflections()
        self.calculate_nose_junction_reflections()

    def calculate_main_tract_reflections(self):
        a = sqr(self.diameter)
        for i in range(1,N):
            self.reflection[i] = self.new_reflection[i]
            sumv = a[i-1]+a[i]
            self.new_reflection[i] = (a[i-1]-a[i])/sumv if abs(sumv)>1e-6 else 1.0

    def calculate_nose_junction_reflections(self):
        self.reflection_left = self.new_reflection_left
        self.reflection_right = self.new_reflection_right
        self.reflection_nose = self.new_reflection_nose
        velum_a = sqr(self.nose_diameter[0])
        an0 = sqr(self.diameter[NOSE_START])
        an1 = sqr(self.diameter[NOSE_START+1])
        sumv = an0 + an1 + velum_a
        if abs(sumv)>1e-6:
            self.new_reflection_left = (2*an0 - sumv)/sumv
            self.new_reflection_right = (2*an1 - sumv)/sumv
            self.new_reflection_nose = (2*velum_a - sumv)/sumv
        else:
            self.new_reflection_left = self.new_reflection_right = self.new_reflection_nose = 1.0

    def step(self, glottal_output: float, lambda_: float) -> float:
        self.process_transients()
        self.add_turbulence_noise()
        self.junction_output_right[0] = self.left[0]*GLOTTAL_REFLECTION + glottal_output
        self.junction_output_left[N] = self.right[N-1]*LIP_REFLECTION
        for i in range(1,N):
            r = interpolate(self.reflection[i], self.new_reflection[i], lambda_)
            w = r*(self.right[i-1]+self.left[i])
            self.junction_output_right[i] = self.right[i-1]-w
            self.junction_output_left[i] = self.left[i]+w
        i = NOSE_START
        r = interpolate(self.reflection_left, self.new_reflection_left, lambda_)
        self.junction_output_left[i] = r*self.right[i-1] + (1+r)*(self.nose_left[0]+self.left[i])
        r = interpolate(self.reflection_right, self.new_reflection_right, lambda_)
        self.junction_output_right[i] = r*self.left[i] + (1+r)*(self.right[i-1]+self.nose_left[0])
        r = interpolate(self.reflection_nose, self.new_reflection_nose, lambda_)
        self.nose_junction_output_right[0] = r*self.nose_left[0] + (1+r)*(self.left[i]+self.right[i-1])
        for i in range(N):
            right = self.junction_output_right[i]*0.999
            left = self.junction_output_left[i+1]*0.999
            self.right[i] = right
            self.left[i] = left
            amplitude = abs(right+left)
            self.max_amplitude[i] = max(self.max_amplitude[i]*0.9999, amplitude)
        lip_output = self.right[N-1]
        self.nose_junction_output_left[NOSE_LEN] = self.nose_right[NOSE_LEN-1]*LIP_REFLECTION
        for i in range(1,NOSE_LEN):
            w = self.nose_reflection[i]*(self.nose_right[i-1]+self.nose_left[i])
            self.nose_junction_output_right[i] = self.nose_right[i-1]-w
            self.nose_junction_output_left[i] = self.nose_left[i]+w
        for i in range(NOSE_LEN):
            right = self.nose_junction_output_right[i]
            left = self.nose_junction_output_left[i+1]
            self.nose_right[i]=right
            self.nose_left[i]=left
            amplitude = abs(right+left)
            self.nose_max_amplitude[i] = max(self.nose_max_amplitude[i]*0.9999, amplitude)
        nose_output = self.nose_right[NOSE_LEN-1]
        self.sample_count += 1
        self.time = self.sample_count/self.sample_rate
        return float(lip_output + nose_output)

    def process_transients(self):
        for idx in reversed(range(len(self.transients))):
            trans = self.transients[idx]
            time_alive = self.time - trans.start_time
            if time_alive > trans.life_time:
                self.transients.pop(idx)
                continue
            amplitude = trans.strength * 2**(-trans.exponent*time_alive)
            self.right[trans.position] += amplitude*0.5
            self.left[trans.position] += amplitude*0.5

    def add_turbulence_noise(self):
        FRICATIVE_ATTACK_TIME = 0.1
        noises = []
        for p in self.turbulence_points:
            if p.position < 2.0 or p.position > N:
                continue
            if p.diameter <= 0:
                continue
            if np.isnan(p.end_time):
                intensity = (self.time - p.start_time)/FRICATIVE_ATTACK_TIME
            else:
                intensity = 1.0 - (self.time - p.end_time)/FRICATIVE_ATTACK_TIME
            intensity = np.clip(intensity,0.0,1.0)
            if intensity<=0:
                continue
            turbulence_noise = 0.66*self.frication_noise_source()*intensity*self.glottis.get_noise_modulator()
            noises.append((turbulence_noise,p.position,p.diameter))
        for noise,pos,diam in noises:
            self.add_turbulence_noise_at_position(noise,pos,diam)

    def add_turbulence_noise_at_position(self, turbulence_noise, position, diameter):
        i = int(np.floor(position))
        delta = position - i
        thinnes0 = np.clip(8*(0.7-diameter),0.0,1.0)
        openness = np.clip(30*(diameter-0.3),0.0,1.0)
        noise0 = turbulence_noise*(1-delta)*thinnes0*openness
        noise1 = turbulence_noise*delta*thinnes0*openness
        if i+1 < N:
            idx = i+1
            self.right[idx] += noise0*0.5
            self.left[idx] += noise0*0.5
        if i+2 < N:
            idx = i+2
            self.right[idx] += noise1*0.5
            self.left[idx] += noise1*0.5
