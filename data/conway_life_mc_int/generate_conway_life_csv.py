#!/usr/bin/env python3
"""Generate an integer CSV of Conway-like cellular automata frames.

The output is intentionally shaped like the roomba grayscale data: frame pixels are
stored as p0, p1, ... integer columns so the shared roomba viewer can display the
sequence, and every column can be converted to a nanoGPT multicontext integer
stream with data/csv_mc_int/get_dataset.sh.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    name: str
    birth: frozenset[int]
    survive: frozenset[int]


RULES = [
    Rule("life_b3_s23", frozenset({3}), frozenset({2, 3})),
    Rule("highlife_b36_s23", frozenset({3, 6}), frozenset({2, 3})),
    Rule("seeds_b2_s", frozenset({2}), frozenset()),
    Rule("daynight_b3678_s34678", frozenset({3, 6, 7, 8}), frozenset({3, 4, 6, 7, 8})),
]

PATTERNS = ["random", "glider", "blinker", "toad", "beacon", "rpentomino", "pulsar"]
PATTERN_CELLS = {
    "glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "blinker": [(0, 0), (0, 1), (0, 2)],
    "toad": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
    "beacon": [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (3, 3)],
    "rpentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "pulsar": [
        (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10),
        (2, 0), (2, 5), (2, 7), (2, 12), (3, 0), (3, 5), (3, 7), (3, 12),
        (4, 0), (4, 5), (4, 7), (4, 12), (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (5, 10),
        (7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10),
        (8, 0), (8, 5), (8, 7), (8, 12), (9, 0), (9, 5), (9, 7), (9, 12),
        (10, 0), (10, 5), (10, 7), (10, 12), (12, 2), (12, 3), (12, 4), (12, 8), (12, 9), (12, 10),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_csv", default="input.csv")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64, help="Frames per episode")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alive_value", type=int, default=255, choices=[1, 255])
    parser.add_argument("--density_min", type=float, default=0.12)
    parser.add_argument("--density_max", type=float, default=0.42)
    parser.add_argument("--mutation_chance", type=float, default=0.015, help="Per-cell random flip probability after each step")
    return parser.parse_args()


def empty_grid(width: int, height: int) -> list[list[int]]:
    return [[0 for _ in range(width)] for _ in range(height)]


def add_pattern(grid: list[list[int]], pattern: str, rng: random.Random) -> None:
    height = len(grid)
    width = len(grid[0])
    cells = PATTERN_CELLS[pattern]
    max_r = max(r for r, _ in cells) + 1
    max_c = max(c for _, c in cells) + 1
    base_r = rng.randrange(height)
    base_c = rng.randrange(width)
    if height >= max_r and width >= max_c:
        base_r = rng.randrange(height - max_r + 1)
        base_c = rng.randrange(width - max_c + 1)
    for r, c in cells:
        grid[(base_r + r) % height][(base_c + c) % width] = 1


def initial_grid(width: int, height: int, density: float, pattern: str, rng: random.Random) -> list[list[int]]:
    grid = empty_grid(width, height)
    if pattern == "random":
        for r in range(height):
            for c in range(width):
                grid[r][c] = 1 if rng.random() < density else 0
    else:
        add_pattern(grid, pattern, rng)
        # Add a light random background so named patterns interact instead of
        # staying perfectly isolated every time.
        for r in range(height):
            for c in range(width):
                if rng.random() < density * 0.20:
                    grid[r][c] ^= 1
    return grid


def neighbor_count(grid: list[list[int]], row: int, col: int) -> int:
    height = len(grid)
    width = len(grid[0])
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += grid[(row + dr) % height][(col + dc) % width]
    return total


def step_grid(grid: list[list[int]], rule: Rule, rng: random.Random, mutation_chance: float) -> list[list[int]]:
    height = len(grid)
    width = len(grid[0])
    nxt = empty_grid(width, height)
    for r in range(height):
        for c in range(width):
            n = neighbor_count(grid, r, c)
            alive = grid[r][c] == 1
            nxt[r][c] = 1 if (n in rule.survive if alive else n in rule.birth) else 0
            if mutation_chance > 0.0 and rng.random() < mutation_chance:
                nxt[r][c] ^= 1
    return nxt


def transitions(prev: list[list[int]] | None, grid: list[list[int]]) -> tuple[int, int]:
    if prev is None:
        return 0, 0
    born = died = 0
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            old = prev[r][c]
            if old == 0 and value == 1:
                born += 1
            elif old == 1 and value == 0:
                died += 1
    return born, died


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.episodes <= 0 or args.steps <= 1:
        raise ValueError("episodes must be positive and steps must be > 1")
    if not 0.0 <= args.density_min <= args.density_max <= 1.0:
        raise ValueError("density bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= args.mutation_chance <= 1.0:
        raise ValueError("mutation chance must be in [0, 1]")

    rng = random.Random(args.seed)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pixel_count = args.width * args.height
    headers = [
        "timestamp", "episode", "step", "width", "height", "rule_id", "pattern_id",
        "density_percent", "mutation_per_mille", "alive_count", "born_count", "died_count",
    ] + [f"p{i}" for i in range(pixel_count)]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        timestamp = 0
        for episode in range(args.episodes):
            rule_id = rng.randrange(len(RULES))
            pattern_id = rng.randrange(len(PATTERNS))
            density = rng.uniform(args.density_min, args.density_max)
            rule = RULES[rule_id]
            grid = initial_grid(args.width, args.height, density, PATTERNS[pattern_id], rng)
            prev = None
            for step in range(args.steps):
                born, died = transitions(prev, grid)
                alive = sum(sum(row) for row in grid)
                pixels = [cell * args.alive_value for row in grid for cell in row]
                writer.writerow([
                    timestamp % 1000,
                    episode,
                    step,
                    args.width,
                    args.height,
                    rule_id,
                    pattern_id,
                    round(density * 100),
                    round(args.mutation_chance * 1000),
                    alive,
                    born,
                    died,
                    *pixels,
                ])
                prev = grid
                grid = step_grid(grid, rule, rng, args.mutation_chance)
                timestamp += 1

    print(f"Wrote {output} with {args.episodes * args.steps} rows and {pixel_count} pixel columns")
    print("Rules:")
    for i, rule in enumerate(RULES):
        print(f"  {i}: {rule.name}")
    print("Patterns:")
    for i, pattern in enumerate(PATTERNS):
        print(f"  {i}: {pattern}")


if __name__ == "__main__":
    main()
