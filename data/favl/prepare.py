"""Create frequency-adjusted validation loss (FAVL) shuffled datasets.

This utility takes an existing nanoGPT binary dataset directory containing
``meta.pkl``, ``train.bin``, and ``val.bin``. It creates several shuffled
variants, each in its own seed-labeled subdirectory, while preserving the token
frequency distribution in each split by shuffling only token order within that
split.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEFAULT_RANDOM_SEEDS = (1729, 271828, 314159)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Build FAVL dataset variants by copying meta.pkl and thoroughly "
            "shuffling uint16 token order inside train.bin and val.bin."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing meta.pkl, train.bin, and val.bin.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where seed-labeled shuffled variants are written (default: this favl folder).",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=len(DEFAULT_RANDOM_SEEDS),
        help="Number of seed variants to create when --seeds is not supplied (default: 3).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit random seeds to use. Overrides --num_seeds.",
    )
    parser.add_argument(
        "--shuffle_rounds",
        type=int,
        default=4,
        help="Number of full in-memory shuffle passes per split for each seed (default: 4).",
    )
    parser.add_argument(
        "--dtype",
        choices=("uint16",),
        default="uint16",
        help="Token dtype in the .bin files. FAVL currently expects uint16 data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing seed output directory if it already exists.",
    )
    return parser.parse_args()


def _resolve_seeds(args):
    if args.seeds is not None:
        if not args.seeds:
            raise ValueError("--seeds must contain at least one seed when provided.")
        return args.seeds
    if args.num_seeds < 1:
        raise ValueError("--num_seeds must be at least 1.")
    if args.num_seeds <= len(DEFAULT_RANDOM_SEEDS):
        return list(DEFAULT_RANDOM_SEEDS[: args.num_seeds])

    seed_sequence = np.random.SeedSequence(DEFAULT_RANDOM_SEEDS)
    generated = seed_sequence.generate_state(
        args.num_seeds - len(DEFAULT_RANDOM_SEEDS), dtype=np.uint32
    )
    return list(DEFAULT_RANDOM_SEEDS) + [int(seed) for seed in generated]


def _validate_input_dir(input_dir):
    required_paths = {
        "meta": input_dir / "meta.pkl",
        "train": input_dir / "train.bin",
        "val": input_dir / "val.bin",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input directory is missing required file(s): {', '.join(missing)}")
    return required_paths


def _load_meta(meta_path):
    with meta_path.open("rb") as meta_file:
        return pickle.load(meta_file)


def _shuffle_uint16_bin(input_path, output_path, seed, shuffle_rounds):
    if input_path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
        raise ValueError(f"{input_path} size is not divisible by uint16 item size.")

    tokens = np.memmap(input_path, dtype=np.uint16, mode="r")
    shuffled = np.array(tokens, dtype=np.uint16, copy=True)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))

    for _ in tqdm(range(shuffle_rounds), desc=f"Shuffling {input_path.name} seed={seed}"):
        rng.shuffle(shuffled)

    shuffled.tofile(output_path)
    return {
        "file": input_path.name,
        "token_count": int(shuffled.size),
        "dtype": "uint16",
        "shuffle_rounds": int(shuffle_rounds),
    }


def _write_variant(paths, output_dir, seed, shuffle_rounds, overwrite):
    variant_dir = output_dir / f"seed_{seed}_shuffle_rounds_{shuffle_rounds}"
    if variant_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{variant_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(paths["meta"], variant_dir / "meta.pkl")
    split_metrics = [
        _shuffle_uint16_bin(paths["train"], variant_dir / "train.bin", seed, shuffle_rounds),
        _shuffle_uint16_bin(paths["val"], variant_dir / "val.bin", seed + 1, shuffle_rounds),
    ]
    metrics = {
        "favl": {
            "source_dir": str(paths["meta"].parent.resolve()),
            "random_seed": int(seed),
            "validation_random_seed": int(seed + 1),
            "shuffle_rounds": int(shuffle_rounds),
            "split_metrics": split_metrics,
        }
    }
    with (variant_dir / "favl_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
    return variant_dir


def main():
    args = parse_arguments()
    if args.shuffle_rounds < 1:
        raise ValueError("--shuffle_rounds must be at least 1.")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = _validate_input_dir(input_dir)
    meta = _load_meta(paths["meta"])
    if meta.get("vocab_size", 0) > np.iinfo(np.uint16).max:
        raise ValueError("FAVL uint16 shuffling requires vocab_size <= 65535 in meta.pkl.")

    output_dir.mkdir(parents=True, exist_ok=True)
    created_dirs = []
    for seed in _resolve_seeds(args):
        created_dirs.append(_write_variant(paths, output_dir, int(seed), args.shuffle_rounds, args.overwrite))

    print("Created FAVL shuffled variants:")
    for directory in created_dirs:
        print(f"- {directory}")


if __name__ == "__main__":
    main()
