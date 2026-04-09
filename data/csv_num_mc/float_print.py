#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Print FP16 (float16) values from a binary file."
    )

    parser.add_argument("file", help="Path to binary file")
    parser.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="Endianness of file (default: little)"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Print index alongside values"
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("File not found:", args.file)
        sys.exit(1)

    filesize = os.path.getsize(args.file)
    if filesize % 2 != 0:
        print("Warning: file size is not multiple of 2 bytes.")

    # Choose dtype with explicit endianness
    if args.endian == "little":
        dtype = np.dtype("<f2")   # little-endian float16
    else:
        dtype = np.dtype(">f2")   # big-endian float16

    data = np.fromfile(args.file, dtype=dtype)

    if args.index:
        for i, val in enumerate(data):
            print(f"{i}: {val}")
    else:
        for val in data:
            print(val)


if __name__ == "__main__":
    main()
