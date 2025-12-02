# MineRL Navigate pixel channel extraction

This folder contains a helper script for generating a pixel-wise channel dataset from the [MineRL Navigate](https://pypi.org/project/minerl-navigate/) video collection. The upstream dataset exposes 64x64 RGB video clips of length 500 frames. The included script downloads the dataset via `tensorflow_datasets`, then writes per-pixel channel arrays into an organized directory structure that is easy to consume for downstream experiments.

## Requirements
- Python 3.9+
- `tensorflow` and `tensorflow-datasets` (required to load the dataset)
- The `minerl-navigate` package (provides the dataset builder used by `tensorflow-datasets`)

Install the dependencies into your environment:

```bash
pip install tensorflow tensorflow-datasets minerl-navigate
```

## Usage
Run the helper script to download the dataset split you care about and emit pixel-level channel arrays:

```bash
python data/minerl_navigate/extract_minerl_navigate_pixels.py \
  --split train \
  --output_dir /path/to/output \
  --max_videos 10
```

Key notes:
- Use `--split` to choose `train` or `test`.
- Use `--max_videos` to limit how many videos are processed (omit to process the entire split).
- By default, the script creates all necessary pixel/channel directories before exporting data.

## Output layout
For each pixel location `(row, col)`, the script creates a directory named `pixel_{row}_{col}` containing channel-specific subfolders:

```
/output_dir
  /pixel_00_00
    /r        # per-video NumPy arrays for the red channel across frames
    /g        # green channel arrays
    /b        # blue channel arrays
    /grayscale # mean of (r, g, b) per frame
  /pixel_00_01
    ...
```

Each saved file is a `.npy` array with shape `(frames,)`, storing the channel intensity over time for a single video. The grayscale values are stored as `float16` (computed as the mean of the three color channels) to reduce disk usage while preserving per-frame intensity information.

## get_dataset.sh
`get_dataset.sh` is provided as a convenience wrapper for environments that prefer shell entry points. Adjust the variables inside the script to point at your desired output directory or override via environment variables when invoking:

```bash
bash data/minerl_navigate/get_dataset.sh
```

This will process the training split and emit pixel-level channel arrays into `./minerl_navigate_output` by default.
