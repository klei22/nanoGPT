# Video Optical Flow Dataset

This dataset prepares a video file by resizing each frame to `n x n` pixels and then
computes optical flow between consecutive frames. For every pixel position `(i, j)`
four datasets are created containing the 8‑bit values of the **R**, **G**, **B** and
**grayscale** channels over time. Additionally, optical flow values are saved for
each pixel.

## Usage

```bash
python3 prepare.py --input path/to/video.mp4 --size 64 --out_dir output
```

This will create the following directory structure in `output`:

```
output/
  r/             # 8-bit values for the red channel
  g/             # 8-bit values for the green channel
  b/             # 8-bit values for the blue channel
  gray/          # 8-bit grayscale values
  flow_x/        # scaled horizontal optical flow values (uint8)
  flow_y/        # scaled vertical optical flow values (uint8)
```

Each directory contains binary files named `i_j.bin` which store the sequence of
values for that pixel across all frames.
