import random


class SimpleSnake:
    """Tiny grid-based snake-style toy environment.

    Actions: 0=left, 1=straight, 2=right (relative to current direction).
    Reward: +1 for eating food, -1 for collisions, -0.01 per step.
    """

    def __init__(self, grid_size=6, max_steps=50, seed=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        center = self.grid_size // 2
        self.head = (center, center)
        self.direction = (0, 1)  # moving right
        self.steps = 0
        self._place_food()
        return self._state()

    def _place_food(self):
        while True:
            x = self.random.randrange(self.grid_size)
            y = self.random.randrange(self.grid_size)
            if (x, y) != self.head:
                self.food = (x, y)
                return

    def _turn(self, action):
        dx, dy = self.direction
        if action == 0:  # left
            return (-dy, dx)
        if action == 2:  # right
            return (dy, -dx)
        return (dx, dy)

    def _state(self):
        return {
            "head": self.head,
            "food": self.food,
            "direction": self.direction,
            "steps": self.steps,
        }

    def step(self, action):
        self.steps += 1
        self.direction = self._turn(action)
        hx, hy = self.head
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.head = new_head
            return -1.0

        self.head = new_head
        reward = -0.01
        if self.head == self.food:
            reward = 1.0
            self._place_food()
        if self.steps >= self.max_steps:
            self.reset()
        return reward


def make_env():
    return SimpleSnake()
