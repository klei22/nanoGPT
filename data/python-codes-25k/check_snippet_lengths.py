#!/usr/bin/env python3
import json
import sys
from typing import Dict, Any

REQUIRED_KEYS = ["output", "general", "param_nesting"]

# Global counters
GLOBAL_SUCCESS = 0
GLOBAL_FAIL = 0
GLOBAL_TOTAL = 0

def load_data(path: str):
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)

def check_entry(entry: Dict[str, Any], idx: int):
    global GLOBAL_FAIL, GLOBAL_SUCCESS, GLOBAL_TOTAL

    print(f"\n=== Checking entry {idx}: {entry.get('name', '(no name)')} ===")

    # Confirm required keys exist
    for key in REQUIRED_KEYS:
        if key not in entry:
            print(f"[ERROR] Missing key '{key}' in entry {idx}. Skipping entry.")
            return

    # Convert to list of lines
    lines = { key: entry[key].split("\n") for key in REQUIRED_KEYS }

    # Validate equal number of lines
    line_counts = { key: len(lines[key]) for key in REQUIRED_KEYS }
    if len(set(line_counts.values())) != 1:
        print("[ERROR] Line count mismatch:")
        for k, v in line_counts.items():
            print(f"  {k}: {v}")
        return

    n = line_counts["output"]
    print(f"Line count: {n}")

    entry_success = 0
    entry_fail = 0

    # Check each line
    for i in range(n):
        GLOBAL_TOTAL += 1

        lengths = { k: len(lines[k][i]) for k in REQUIRED_KEYS }

        if len(set(lengths.values())) != 1:
            entry_fail += 1
            GLOBAL_FAIL += 1

            print(f"\n[MISMATCH at line {i}]")
            for k in REQUIRED_KEYS:
                print(f"  {k} ({lengths[k]} chars): {repr(lines[k][i])}")
        else:
            entry_success += 1
            GLOBAL_SUCCESS += 1

    # Entry-level summary
    print(f"\nEntry {idx} summary:")
    print(f"  ✔ Success: {entry_success}")
    print(f"  ✖ Failures: {entry_fail}")
    if entry_success + entry_fail > 0:
        pct = 100 * (entry_success / (entry_success + entry_fail))
        print(f"  ✅ Success rate: {pct:.2f}%")

def print_final_report():
    print("\n=======================")
    print(" FINAL CONSISTENCY REPORT")
    print("=======================\n")

    total = GLOBAL_TOTAL
    success = GLOBAL_SUCCESS
    fail = GLOBAL_FAIL

    print(f"Total line checks performed: {total}")
    print(f"  ✔ Successful: {success}")
    print(f"  ✖ Failed:     {fail}")

    if total > 0:
        pct_success = 100 * (success / total)
        pct_fail = 100 * (fail / total)
        print(f"\nSuccess rate: {pct_success:.2f}%")
        print(f"Failure rate: {pct_fail:.2f}%")
    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_snippet_lengths.py <file.json|file.yaml>")
        sys.exit(1)

    data = load_data(sys.argv[1])

    if not isinstance(data, list):
        print("[ERROR] Expected a top-level list.")
        sys.exit(1)

    for i, entry in enumerate(data):
        check_entry(entry, i)

    print_final_report()

if __name__ == "__main__":
    main()

