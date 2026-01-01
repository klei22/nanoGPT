#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Whisper-style mel CSV files.")
    parser.add_argument("csv_path", type=Path, help="Path to the mel CSV file.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save the plot (PNG).")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for color scale.")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for color scale.")
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap name.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.loadtxt(args.csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]

    plt.figure(figsize=(10, 4))
    plt.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    plt.xlabel("Frame")
    plt.ylabel("Mel channel")
    if args.title:
        plt.title(args.title)
    plt.colorbar(label="Value")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
