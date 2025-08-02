#!/usr/bin/env python3
"""Demonstration script for multidataset_wte training.

This script downloads and tokenizes the `databricks-dolly-15k` and
`shakespeare_char` datasets using character tokenization. After the
datasets are prepared it launches training in multidataset mode with
separate embeddings for each dataset and prints samples with
``max_sample_tokens`` set to 128.

This roughly mirrors the commands that would be executed manually from
the shell but bundles them in one Python file for convenience.
"""

import os
import subprocess
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent


def run(cmd: str, cwd: Path | None = None) -> None:
    """Run a shell command and stream output."""
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def prepare_dolly() -> None:
    data_dir = REPO_DIR / "data" / "databricks-dolly-15k"
    run("bash get_dataset.sh", cwd=data_dir)
    run("python3 prepare.py --method char -t input.txt", cwd=data_dir)


def prepare_shakespeare() -> None:
    data_dir = REPO_DIR / "data" / "shakespeare_char"
    run("bash get_dataset.sh", cwd=data_dir)


def train_model() -> None:
    cmd = (
        "python3 train.py "
        "--training_mode multidataset "
        "--dataset shakespeare_char "
        "--dataset_list shakespeare_char databricks-dolly-15k "
        "--dataset_sampling_probs 1 1 "
        "--multidataset_wte "
        "--max_sample_tokens 128"
    )
    run(cmd, cwd=REPO_DIR)


def main() -> None:
    prepare_dolly()
    prepare_shakespeare()
    train_model()


if __name__ == "__main__":
    main()
