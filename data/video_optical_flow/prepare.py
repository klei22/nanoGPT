import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Process video into per-pixel datasets with optical flow")
    p.add_argument("--input", required=True, help="Path to input video")
    p.add_argument("--size", type=int, default=64, help="Resize frames to NxN")
    p.add_argument("--out_dir", default="output", help="Directory to store dataset")
    return p.parse_args()


def ensure_dirs(base, names):
    dirs = {}
    for name in names:
        d = Path(base) / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs


def save_pixel_data(data_map, out_dirs, dtype=np.uint8):
    for key, dirs in out_dirs.items():
        for i, row in enumerate(data_map[key]):
            for j, values in enumerate(row):
                arr = np.asarray(values, dtype=dtype)
                (dirs / f"{i}_{j}.bin").write_bytes(arr.tobytes())


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {args.input}")

    out_dirs = ensure_dirs(args.out_dir, ["r", "g", "b", "gray", "flow_x", "flow_y"])

    pixel_data = {
        "r": [[[] for _ in range(args.size)] for _ in range(args.size)],
        "g": [[[] for _ in range(args.size)] for _ in range(args.size)],
        "b": [[[] for _ in range(args.size)] for _ in range(args.size)],
        "gray": [[[] for _ in range(args.size)] for _ in range(args.size)],
    }
    flow_data_x = [[[] for _ in range(args.size)] for _ in range(args.size)]
    flow_data_y = [[[] for _ in range(args.size)] for _ in range(args.size)]

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Video contains no frames")
    prev = cv2.resize(prev, (args.size, args.size))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (args.size, args.size))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]

        for i in range(args.size):
            for j in range(args.size):
                pixel_data["r"][i][j].append(r[i, j])
                pixel_data["g"][i][j].append(g[i, j])
                pixel_data["b"][i][j].append(b[i, j])
                pixel_data["gray"][i][j].append(gray[i, j])
                flow_data_x[i][j].append(fx[i, j])
                flow_data_y[i][j].append(fy[i, j])

        prev_gray = gray

    cap.release()

    # scale optical flow to uint8
    fx_all = np.concatenate([np.concatenate(row) for row in flow_data_x])
    fy_all = np.concatenate([np.concatenate(row) for row in flow_data_y])
    fx_min, fx_max = fx_all.min(), fx_all.max()
    fy_min, fy_max = fy_all.min(), fy_all.max()

    def scale(values, vmin, vmax):
        scaled = 255 * (np.asarray(values) - vmin) / (vmax - vmin + 1e-9)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    for i in range(args.size):
        for j in range(args.size):
            flow_data_x[i][j] = scale(flow_data_x[i][j], fx_min, fx_max)
            flow_data_y[i][j] = scale(flow_data_y[i][j], fy_min, fy_max)

    save_pixel_data(pixel_data, {k: out_dirs[k] for k in ["r", "g", "b", "gray"]})
    save_pixel_data({"flow_x": flow_data_x, "flow_y": flow_data_y},
                    {"flow_x": out_dirs["flow_x"], "flow_y": out_dirs["flow_y"]})


if __name__ == "__main__":
    main()
