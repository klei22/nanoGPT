#!/bin/bash
set -euo pipefail

# Downloader and formatter for the English-to-Zh/Ko/Ja splits of
# https://huggingface.co/datasets/haoranxu/flores-200
#
# Usage:
#   bash get_dataset.sh
#
# Requirements:
#   pip install huggingface_hub pandas pyarrow
#
# Output:
#   - input.txt with alternating prompt/response blocks per example
#     using the same prefixes as the dataset template (#U for source,
#     #B for target).

python - <<'PY'
from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path

REPO_ID = "haoranxu/flores-200"
SPLITS = [
    ("en-zh", "zh"),
    ("en-ko", "ko"),
    ("en-ja", "ja"),
]
OUTPUT_PATH = Path("input.txt")

OUTPUT_PATH.write_text("")

for split, target_key in SPLITS:
    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{split}/test-00000-of-00001.parquet",
        repo_type="dataset",
    )

    df = pd.read_parquet(parquet_path)

    # Each cell is a dict like {"en": "...", "zh": "..."}
    for record in df[split]:
        source = record.get("en", "").strip()
        target = record.get(target_key, "").strip()
        if not source or not target:
            continue

        with OUTPUT_PATH.open("a") as f:
            f.write(f"#U:\n{source}\n")
            f.write(f"#B:\n{target}\n")
PY

printf "\nFinished writing $(wc -l < input.txt) lines to input.txt\n"
