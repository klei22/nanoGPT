"""Export pixel-wise channel arrays from the MineRL Navigate video dataset.

Each video in the dataset has shape (frames, height, width, channels) where
height and width are 64 and channels correspond to RGB values in uint8.
This script downloads the dataset archive directly (no TensorFlow
dependency), then writes per-pixel channel sequences to disk. The
layout looks like:

output_dir/
  pixel_00_00/
    r/train_video_00000.npy        # uint8 array of shape (frames,)
    g/train_video_00000.npy
    b/train_video_00000.npy
    grayscale/train_video_00000.npy  # float16 array of shape (frames,)
  pixel_00_01/
    ...

The grayscale values are the arithmetic mean of the RGB channels stored as float16.
"""

from __future__ import annotations

import argparse
import pathlib
import urllib.request
import zipfile
from typing import Iterable, Tuple

import imageio.v3 as iio
import numpy as np

DOWNLOAD_URL = "https://archive.org/download/minerl_navigate/minerl_navigate.zip"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which dataset split to export.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=pathlib.Path,
        help="Directory where per-pixel channel arrays will be stored.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache" / "minerl_navigate",
        help=(
            "Directory to store or locate the extracted minerl_navigate archive. "
            "If the train/test folders are missing, the script will download the archive."
        ),
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional limit on how many videos to process from the split.",
    )
    return parser.parse_args()


def ensure_pixel_directories(output_dir: pathlib.Path, height: int, width: int) -> None:
    """Create folders for every pixel and channel.

    Layout example for pixel row=0, col=1:
    output_dir/pixel_00_01/{r,g,b,grayscale}
    """
    for row in range(height):
        for col in range(width):
            pixel_root = output_dir / f"pixel_{row:02d}_{col:02d}"
            for channel in ("r", "g", "b", "grayscale"):
                (pixel_root / channel).mkdir(parents=True, exist_ok=True)


def ensure_dataset(dataset_dir: pathlib.Path) -> pathlib.Path:
    """Download and extract the MineRL Navigate archive if needed.

    Returns the path containing `train/` and `test/` subdirectories with MP4 files.
    """

    dataset_dir = dataset_dir.expanduser()
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    if train_dir.exists() and test_dir.exists():
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir / "minerl_navigate.zip"

    if not zip_path.exists():
        print(f"Downloading dataset archive to {zip_path}...")
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)

    print(f"Extracting archive at {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    extracted_root = dataset_dir / "minerl_navigate"
    if extracted_root.exists():
        return extracted_root

    return dataset_dir


def iter_dataset(dataset_dir: pathlib.Path, split: str) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield index and video numpy array for the requested split."""

    split_dir = dataset_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory {split_dir} not found. Ensure the dataset archive was extracted."
        )

    for idx, video_path in enumerate(sorted(split_dir.glob("*.mp4"))):
        yield idx, iio.imread(video_path)


def export_video(
    video_idx: int, video: np.ndarray, output_dir: pathlib.Path, split: str
) -> None:
    """Write per-pixel channel arrays for a single video."""
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected video of shape (frames, H, W, 3), got {video.shape}")

    frames, height, width, _ = video.shape
    file_stem = f"{split}_video_{video_idx:05d}"

    grayscale_frames = video.mean(axis=-1, dtype=np.float32).astype(np.float16)

    for row in range(height):
        for col in range(width):
            pixel_series = video[:, row, col, :]
            pixel_root = output_dir / f"pixel_{row:02d}_{col:02d}"

            np.save(pixel_root / "r" / f"{file_stem}.npy", pixel_series[:, 0])
            np.save(pixel_root / "g" / f"{file_stem}.npy", pixel_series[:, 1])
            np.save(pixel_root / "b" / f"{file_stem}.npy", pixel_series[:, 2])
            np.save(
                pixel_root / "grayscale" / f"{file_stem}.npy",
                grayscale_frames[:, row, col],
            )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MineRL Navigate {args.split} split...")
    dataset_root = ensure_dataset(args.dataset_dir)

    dataset_iter = iter_dataset(dataset_root, args.split)

    # Inspect shape from first element to avoid creating 4096 directories twice.
    first_item = next(dataset_iter, None)
    if first_item is None:
        raise RuntimeError(f"No examples found for split '{args.split}'")

    first_idx, first_video = first_item
    _, height, width, _ = first_video.shape
    ensure_pixel_directories(args.output_dir, height, width)

    print(f"Created pixel folders under {args.output_dir} for {height}x{width} frames")

    export_video(first_idx, first_video, args.output_dir, args.split)

    processed = 1
    for video_idx, video in dataset_iter:
        if args.max_videos is not None and processed >= args.max_videos:
            break
        export_video(video_idx, video, args.output_dir, args.split)
        processed += 1

    print(f"Finished exporting {processed if args.max_videos is None else min(processed, args.max_videos)} videos.")


if __name__ == "__main__":
    main()
