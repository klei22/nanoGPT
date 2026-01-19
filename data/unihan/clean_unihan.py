import re
from tqdm import tqdm

INPUT = "unihan.yaml"
OUTPUT = "unihan_nonull.yaml"

# Regex patterns for null-like fields
NULL_PATTERNS = [
    r": null\s*$",
    r": None\s*$",
    r": ''\s*$",
    r": \"\"\s*$",
]

combined = re.compile("|".join(NULL_PATTERNS))

# Count lines for progress bar
with open(INPUT, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, total=total_lines, desc="Removing null fields"):
        # If this line matches any null pattern, skip it
        if combined.search(line):
            continue
        fout.write(line)

print(f"\nDone! Cleaned file written to {OUTPUT}")

