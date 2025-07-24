from dataclasses import dataclass

@dataclass
class Transient:
    position: int
    start_time: float
    life_time: float
    strength: float
    exponent: float
