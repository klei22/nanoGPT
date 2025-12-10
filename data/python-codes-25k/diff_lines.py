#!/usr/bin/env python3
import sys

def compare_files(f1, f2):
    with open(f1, "r", encoding="utf-8") as a, open(f2, "r", encoding="utf-8") as b:
        lines_a = a.readlines()
        lines_b = b.readlines()

    max_lines = max(len(lines_a), len(lines_b))

    for i in range(max_lines):
        # Handle missing lines
        if i >= len(lines_a):
            print(f"Line {i+1}: FILE1 missing; FILE2 has {len(lines_b[i].rstrip())} chars.")
            continue
        if i >= len(lines_b):
            print(f"Line {i+1}: FILE2 missing; FILE1 has {len(lines_a[i].rstrip())} chars.")
            continue

        la = len(lines_a[i].rstrip("\n"))
        lb = len(lines_b[i].rstrip("\n"))

        if la != lb:
            print(f"Line {i+1} differs: file1={la} chars, file2={lb} chars")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_line_lengths.py file1.txt file2.txt")
        sys.exit(1)

    compare_files(sys.argv[1], sys.argv[2])

