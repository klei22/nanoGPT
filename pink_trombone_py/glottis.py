import numpy as np
from dataclasses import dataclass
from .noise_gen import NoiseGenerator
from .noise import new_filtered_noise_source, NoiseSource
from .math_utils import interpolate

PI = np.pi

@dataclass
class Glottis:
    sample_rate: int
    noise_source: NoiseSource
    seed: int
    always_voice: bool = True
    auto_wobble: bool = True
    target_tenseness: float = 0.6
    target_frequency: float = 140.0
    vibrato_amount: float = 0.005
    vibrato_frequency: float = 6.0

    def __post_init__(self):
        self.noise_generator = NoiseGenerator(self.seed)
        self.sample_count = 0
        self.intensity = 0.0
        self.loudness = 1.0
        self.smooth_frequency = 140.0
        self.time_in_waveform = 0.0
        self.old_tenseness = 0.6
        self.new_tenseness = 0.6
        self.old_frequency = 140.0
        self.new_frequency = 140.0
        self.aspiration_noise_source = new_filtered_noise_source(
            500.0, 0.5, self.sample_rate, 0x8000, self.noise_source
        )
        self.waveform_length = 0.0
        self.alpha = 0.0
        self.e0 = 0.0
        self.epsilon = 0.0
        self.shift = 0.0
        self.delta = 0.0
        self.te = 0.0
        self.omega = 0.0
        self.setup_waveform(0.0)

    def set_musical_note(self, semitone: float):
        A4 = 440.0
        self.target_frequency = A4 * (2.0 ** (semitone/12.0))

    def step(self, lambda_: float) -> float:
        time = self.sample_count / self.sample_rate
        if self.time_in_waveform > self.waveform_length:
            self.time_in_waveform -= self.waveform_length
            self.setup_waveform(lambda_)
        out1 = self.normalized_lf_waveform(self.time_in_waveform / self.waveform_length)
        asp_noise = self.aspiration_noise_source()
        aspiration1 = self.intensity * (1.0 - np.sqrt(self.target_tenseness)) * self.get_noise_modulator() * asp_noise
        aspiration2 = aspiration1 * (0.2 + 0.02 * self.noise_generator.simplex(time*1.99))
        result = out1 + aspiration2
        self.sample_count += 1
        self.time_in_waveform += 1.0 / self.sample_rate
        return result

    def get_noise_modulator(self) -> float:
        voiced = 0.1 + 0.2 * max(0.0, np.sin(PI*2.0*self.time_in_waveform/self.waveform_length))
        return self.target_tenseness * self.intensity * voiced + (1.0 - self.target_tenseness*self.intensity)*0.3

    def adjust_parameters(self, delta_time: float):
        delta = delta_time * self.sample_rate / 512.0
        old_time = self.sample_count / self.sample_rate
        new_time = old_time + delta_time
        self.adjust_intensity(delta)
        self.calculate_new_frequency(new_time, delta)
        self.calculate_new_tenseness(new_time)

    def calculate_new_frequency(self, time: float, delta_time: float):
        if self.intensity == 0.0:
            self.smooth_frequency = self.target_frequency
        elif self.target_frequency > self.smooth_frequency:
            self.smooth_frequency = min(self.target_frequency, self.smooth_frequency*(1.0+0.1*delta_time))
        elif self.target_frequency < self.smooth_frequency:
            self.smooth_frequency = max(self.target_frequency, self.smooth_frequency/(1.0+0.1*delta_time))
        self.old_frequency = self.new_frequency
        self.new_frequency = max(10.0, self.smooth_frequency * (1.0 + self.calculate_vibrato(time)))

    def calculate_new_tenseness(self, time: float):
        self.old_tenseness = self.new_tenseness
        self.new_tenseness = self.target_tenseness + 0.1*self.noise_generator.simplex(time*0.46) + 0.05*self.noise_generator.simplex(time*0.36)
        self.new_tenseness = max(0.0, self.new_tenseness)
        if self.always_voice:
            self.new_tenseness += (3.0 - self.target_tenseness) * (1.0 - self.intensity)

    def adjust_intensity(self, delta: float):
        self.intensity += 0.13*delta
        self.intensity = max(0.0, min(1.0, self.intensity))

    def calculate_vibrato(self, time: float) -> float:
        vibrato = self.vibrato_amount * np.sin(PI*2.0*time*self.vibrato_frequency)
        vibrato += 0.02*self.noise_generator.simplex(time*4.07)
        vibrato += 0.04*self.noise_generator.simplex(time*2.15)
        if self.auto_wobble:
            vibrato += 0.2*self.noise_generator.simplex(time*0.96)
            vibrato += 0.4*self.noise_generator.simplex(time*0.5)
        return vibrato

    def setup_waveform(self, lambda_: float):
        frequency = interpolate(self.old_frequency, self.new_frequency, lambda_)
        tenseness = interpolate(self.old_tenseness, self.new_tenseness, lambda_)
        self.waveform_length = 1.0 / frequency
        self.loudness = max(0.0, tenseness)**0.25
        rd = np.clip(3.0*(1.0 - tenseness), 0.5, 2.7)
        ra = -0.01 + 0.048*rd
        rk = 0.224 + 0.118*rd
        rg = (rk/4.0)*(0.5+1.2*rk)/(0.11*rd - ra*(0.5+1.2*rk))
        ta = ra
        tp = 1.0/(2.0*rg)
        te = tp + tp*rk
        epsilon = 1.0/ta
        shift = np.exp(-epsilon*(1.0-te))
        delta = 1.0 - shift
        rhs_integral = ((1.0/epsilon)*(shift-1.0)+(1.0-te)*shift)/delta
        total_lower_integral = rhs_integral - (te - tp)/2.0
        total_upper_integral = -total_lower_integral
        omega = PI/tp
        s = np.sin(omega*te)
        y = -PI*s*total_upper_integral/(tp*2.0)
        z = np.log(y)
        alpha = z/(tp/2.0 - te)
        e0 = -1.0/(s*np.exp(alpha*te))
        self.alpha = alpha
        self.e0 = e0
        self.epsilon = epsilon
        self.shift = shift
        self.delta = delta
        self.te = te
        self.omega = omega

    def normalized_lf_waveform(self, t: float) -> float:
        if t > self.te:
            output = -(np.exp(-self.epsilon*(t-self.te))-self.shift)/self.delta
        else:
            output = self.e0*np.exp(self.alpha*t)*np.sin(self.omega*t)
        return output*self.intensity*self.loudness
