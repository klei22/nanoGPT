import numpy as np
from dataclasses import dataclass
from .tract import Tract, BLADE_START, TIP_START, LIP_START
from .math_utils import move_towards
from .transient import Transient

GRID_OFFSET = 1.7
MOVEMENT_SPEED = 15.0

@dataclass
class TractShaper:
    tract: Tract
    velum_open_target: float = 0.4
    velum_closed_target: float = 0.01
    velum_target: float = 0.0
    tongue_index: float = 12.9
    tongue_diameter: float = 2.43
    last_obstruction: int = -1

    def __post_init__(self):
        self.target_diameter = np.zeros(self.tract.diameter.shape)
        self.shape_noise(True)
        self.tract.calculate_nose_reflections()
        self.shape_noise(False)
        self.shape_main_tract()

    def shape_main_tract(self):
        for i in range(Tract.N):
            d = self.get_rest_diameter(i)
            self.tract.diameter[i] = d
            self.target_diameter[i] = d

    def get_rest_diameter(self, i: int) -> float:
        if i < 7:
            return 0.6
        if i < BLADE_START:
            return 1.1
        if i >= LIP_START:
            return 1.5
        t = 1.1*np.pi*(self.tongue_index - i)/(TIP_START - BLADE_START)
        fixed = 2.0 + (self.tongue_diameter - 2.0)/1.5
        curve = (1.5 - fixed + GRID_OFFSET) * np.cos(t)
        if i == BLADE_START - 2 or i == LIP_START - 1:
            curve *= 0.8
        if i == BLADE_START or i == LIP_START - 2:
            curve *= 0.94
        return 1.5 - curve

    def adjust_tract_shape(self, delta_time: float):
        amount = delta_time*MOVEMENT_SPEED
        new_last = -1
        for i in range(Tract.N):
            diameter = self.tract.diameter[i]
            target = self.target_diameter[i]
            if diameter <= 0.0:
                new_last = i
            if i < Tract.NOSE_START:
                slow_return = 0.6
            elif i >= TIP_START:
                slow_return = 1.0
            else:
                slow_return = 0.6 + 0.4*(i-Tract.NOSE_START)/(TIP_START-Tract.NOSE_START)
            self.tract.diameter[i] = move_towards(diameter,target,slow_return*amount,2.0*amount)
        if self.last_obstruction >=0 and new_last <0 and self.tract.nose_diameter[0] < 0.223:
            self.add_transient(self.last_obstruction)
        self.last_obstruction = new_last
        self.tract.nose_diameter[0] = move_towards(self.tract.nose_diameter[0],self.velum_target,amount*0.25,amount*0.1)

    def add_transient(self, position:int):
        self.tract.transients.append(Transient(position,start_time=self.tract.time,life_time=0.2,strength=0.3,exponent=200.0))

    def shape_noise(self, velum_open: bool):
        self.set_velum_open(velum_open)
        for i in range(Tract.NOSE_LEN):
            d = i*2.0/Tract.NOSE_LEN
            if i==0:
                diameter = self.velum_target
            elif d <1.0:
                diameter = 0.4 + 1.6*d
            else:
                diameter = 0.5 + 1.5*(2.0-d)
            diameter = min(diameter,1.9)
            self.tract.nose_diameter[i] = diameter

    def set_velum_open(self, velum_open: bool):
        self.velum_target = self.velum_open_target if velum_open else self.velum_closed_target
