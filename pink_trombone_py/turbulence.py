from dataclasses import dataclass
import math

@dataclass
class TurbulencePoint:
    diameter: float = 0.0
    position: float = 0.0
    start_time: float = 0.0
    end_time: float = math.nan
