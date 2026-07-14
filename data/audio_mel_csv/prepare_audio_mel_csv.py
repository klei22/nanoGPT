#!/usr/bin/env python3
"""Decompose one audio file or a folder of audio files into quantized mel CSV."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".wave", ".mp3", ".flac", ".ogg", ".m4a"}


def resolve_audio_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    return [path]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="Input audio file or folder containing wav/mp3/flac/ogg/m4a files.")
    p.add_argument("--output_root", default="audio_mel_csv", help="Folder under data/ for datasets and manifest.")
    p.add_argument("--work_dir", default="data/audio_mel_csv/work")
    p.add_argument("--quantized_csv", default=None)
    p.add_argument("--scale", type=int, default=32767, help="Scale normalized mel floats to integer range.")
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--keep_per_file_float_csv", action="store_true", help="Keep intermediate per-audio mel CSV files.")
    args = p.parse_args()

    import numpy as np

    repo = Path(__file__).resolve().parents[2]
    source = Path(args.audio)
    if not source.is_absolute():
        source = repo / source
    audio_files = resolve_audio_inputs(source)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found at {source}")

    work = Path(args.work_dir)
    work = work if work.is_absolute() else repo / work
    work.mkdir(parents=True, exist_ok=True)
    per_file_dir = work / "per_file_float_mels"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    float_csv = work / "mel_float.csv"
    q_csv = Path(args.quantized_csv) if args.quantized_csv else work / "mel_int.csv"
    if not q_csv.is_absolute():
        q_csv = repo / q_csv

    prep = repo / "data/template/prepare.py"
    frames = []
    segments = []
    cursor = 0
    for idx, audio_file in enumerate(audio_files):
        part_csv = per_file_dir / f"{idx:04d}_{audio_file.stem}.csv"
        subprocess.run(
            [
                sys.executable,
                str(prep),
                "--method",
                "whisper_mel_csv",
                "--train_input",
                str(audio_file),
                "--train_output",
                str(part_csv),
                "--percentage_train",
                "1.0",
            ],
            check=True,
            cwd=repo,
        )
        mel_part = np.loadtxt(part_csv, delimiter=",", dtype=np.float32)
        if mel_part.ndim == 1:
            mel_part = mel_part[np.newaxis, :]
        frames.append(mel_part)
        segments.append({"path": str(audio_file), "start_frame": cursor, "frames": int(mel_part.shape[0])})
        cursor += int(mel_part.shape[0])
        if not args.keep_per_file_float_csv:
            part_csv.unlink(missing_ok=True)

    mel = np.vstack(frames)
    np.savetxt(float_csv, mel, delimiter=",", fmt="%.6f")
    q = np.rint(np.clip(mel, 0.0, 1.0) * args.scale).astype(np.int32)
    q_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [f"mel_{i:03d}" for i in range(q.shape[1])]
    with q_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(q.tolist())

    out_root = args.output_root
    subprocess.run(
        [
            sys.executable,
            str(repo / "data/csv_mc_int/prepare_csv_integer_multicontext.py"),
            "--input_csv",
            str(q_csv),
            "--output_root",
            out_root,
            "--train_ratio",
            str(args.train_ratio),
            "--default_range",
            f"0:{args.scale}",
            "--allow_out_of_range",
        ],
        check=True,
        cwd=repo,
    )
    manifest = repo / "data" / out_root / "manifest.json"
    with (repo / "meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m.update(
        {
            "audio_source": str(source),
            "audio_files": [str(p) for p in audio_files],
            "audio_segments": segments,
            "float_mel_csv": str(float_csv),
            "quantized_mel_csv": str(q_csv),
            "quantization": {"scale": args.scale, "inverse": "float = integer / scale"},
            "mel_meta": meta,
        }
    )
    manifest.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"Audio files: {len(audio_files)}")
    print(f"Quantized mel CSV: {q_csv}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
