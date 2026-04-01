"""
Simple Roomba simulation on a 2D grid.

The Roomba navigates a rectangular room, avoiding obstacles,
and attempts to cover as much floor area as possible.
"""

import random
import argparse


class Room:
    """A rectangular room with optional obstacles."""

    EMPTY = 0
    OBSTACLE = 1
    CLEANED = 2

    def __init__(self, width, height, obstacle_fraction=0.1):
        self.width = width
        self.height = height
        self.grid = [[self.EMPTY] * width for _ in range(height)]
        self._place_obstacles(obstacle_fraction)

    def _place_obstacles(self, fraction):
        num_obstacles = int(self.width * self.height * fraction)
        placed = 0
        while placed < num_obstacles:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.grid[y][x] == self.EMPTY:
                self.grid[y][x] = self.OBSTACLE
                placed += 1

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x, y):
        return self.is_valid(x, y) and self.grid[y][x] != self.OBSTACLE

    def clean(self, x, y):
        if self.is_passable(x, y):
            self.grid[y][x] = self.CLEANED

    def coverage(self):
        total = sum(1 for row in self.grid for c in row if c != self.OBSTACLE)
        cleaned = sum(1 for row in self.grid for c in row if c == self.CLEANED)
        return cleaned / total if total > 0 else 0.0


class Roomba:
    """A simple Roomba that moves randomly, bouncing off walls/obstacles."""

    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W

    def __init__(self, room, start_x=0, start_y=0):
        self.room = room
        self.x = start_x
        self.y = start_y
        self.direction = random.randint(0, 3)
        self.room.clean(self.x, self.y)

    def step(self):
        """Move one step. If blocked, pick a new random direction."""
        dx, dy = self.DIRECTIONS[self.direction]
        nx, ny = self.x + dx, self.y + dy

        if self.room.is_passable(nx, ny):
            self.x, self.y = nx, ny
        else:
            self.direction = random.randint(0, 3)
            dx, dy = self.DIRECTIONS[self.direction]
            nx, ny = self.x + dx, self.y + dy
            if self.room.is_passable(nx, ny):
                self.x, self.y = nx, ny

        self.room.clean(self.x, self.y)


def render(room, roomba):
    """Render the room as ASCII art."""
    symbols = {Room.EMPTY: '.', Room.OBSTACLE: '#', Room.CLEANED: ' '}
    lines = []
    for y in range(room.height):
        row = ''
        for x in range(room.width):
            if x == roomba.x and y == roomba.y:
                row += 'R'
            else:
                row += symbols[room.grid[y][x]]
        lines.append(row)
    return '\n'.join(lines)


def run_simulation(width, height, steps, obstacle_fraction, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)

    room = Room(width, height, obstacle_fraction)
    start_x = random.randint(0, width - 1)
    start_y = random.randint(0, height - 1)
    while not room.is_passable(start_x, start_y):
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)

    roomba = Roomba(room, start_x, start_y)

    for i in range(steps):
        roomba.step()
        if verbose and (i + 1) % (steps // 10 or 1) == 0:
            pct = room.coverage() * 100
            print(f"Step {i + 1}/{steps} — Coverage: {pct:.1f}%")

    if verbose:
        print()
        print(render(room, roomba))
        print()

    final_coverage = room.coverage() * 100
    print(f"Final coverage: {final_coverage:.1f}% after {steps} steps")
    return final_coverage


def main():
    parser = argparse.ArgumentParser(description="Roomba Simulation")
    parser.add_argument("--width", type=int, default=20, help="Room width")
    parser.add_argument("--height", type=int, default=20, help="Room height")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--obstacles", type=float, default=0.1, help="Obstacle fraction (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Show progress and final grid")
    args = parser.parse_args()

    run_simulation(args.width, args.height, args.steps, args.obstacles, args.seed, args.verbose)


if __name__ == "__main__":
    main()
