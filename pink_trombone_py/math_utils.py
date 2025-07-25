import numpy as np

def interpolate(i0, i1, v):
    return i0 + v * (i1 - i0)

def move_towards(current: float, target: float, amount_up: float, amount_down: float) -> float:
    if current < target:
        return min(target, current + amount_up)
    else:
        return max(target, current - amount_down)

def sqr(x):
    return x * x
