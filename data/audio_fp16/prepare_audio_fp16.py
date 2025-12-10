import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - guidance utility
    raise SystemExit("numpy is required to run this script. Install with `pip install numpy`." ) from exc

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - guidance utility
    raise SystemExit(
        "soundfile is required to load audio. Install with `pip install soundfile`."
    ) from exc


@dataclass
class ClipInfo:
    path: Path
    num_samples: int
    offset: int
    split: str


SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert audio to fp16-packed train/val bin files.")
    parser.add_argument("--inputs", nargs="+", help="List of audio files to include (wav/flac/ogg/mp3/etc.)")
    parser.add_argument("--output_dir", default="data/audio_fp16", help="Folder to write bin/meta outputs")
    parser.add_argument("--target_sample_rate", type=int, default=16000, help="Target sample rate for resampling")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Portion of clips reserved for validation")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True, help="Normalize audio to [-1, 1]")
    return parser.parse_args()


def linear_resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Minimal linear resampler to avoid heavy dependencies."""
    if orig_sr == target_sr:
        return waveform
    orig_len = waveform.shape[0]
    target_len = int(round(orig_len * target_sr / orig_sr))
    orig_positions = np.linspace(0, orig_len - 1, num=orig_len, dtype=np.float64)
    target_positions = np.linspace(0, orig_len - 1, num=target_len, dtype=np.float64)
    return np.interp(target_positions, orig_positions, waveform).astype(waveform.dtype)


def load_audio(path: Path, target_sr: int, normalize: bool) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = linear_resample(data, sr, target_sr)
        sr = target_sr
    if normalize:
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak
    return data.astype(np.float16), sr


def gather_inputs(inputs: Iterable[str]) -> List[Path]:
    paths = []
    for item in inputs:
        candidate = Path(item)
        if candidate.is_dir():
            for ext in SUPPORTED_EXTS:
                paths.extend(candidate.rglob(f"*{ext}"))
        else:
            paths.append(candidate)
    unique_paths = []
    seen = set()
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        unique_paths.append(path)
    return sorted(unique_paths)


def save_uint16_bin(values: np.ndarray, output_path: Path) -> None:
    values.astype(np.uint16).tofile(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processed_clips").mkdir(parents=True, exist_ok=True)

    audio_paths = gather_inputs(args.inputs)
    if not audio_paths:
        raise SystemExit("No audio files found matching the requested inputs.")

    clip_infos: List[ClipInfo] = []
    train_cursor = 0
    val_cursor = 0

    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    with open(train_bin, "wb") as train_f, open(val_bin, "wb") as val_f:
        for idx, audio_path in enumerate(audio_paths):
            waveform, sr = load_audio(audio_path, args.target_sample_rate, args.normalize)
            uint16_view = waveform.view(np.uint16)

            clip_name = audio_path.stem
            npy_path = output_dir / "processed_clips" / f"{clip_name}.npy"
            np.save(npy_path, waveform)

            split = "val" if (idx / len(audio_paths)) >= (1.0 - args.val_fraction) else "train"
            if split == "train":
                offset = train_cursor
                train_f.write(uint16_view.tobytes())
                train_cursor += uint16_view.size
            else:
                offset = val_cursor
                val_f.write(uint16_view.tobytes())
                val_cursor += uint16_view.size

            clip_infos.append(ClipInfo(audio_path, uint16_view.size, offset, split))

    meta = {
        "tokenizer": "audio_fp16",
        "vocab_size": 65536,
        "sample_rate": args.target_sample_rate,
        "value_dtype": "float16",
        "storage_dtype": "uint16",
        "clips": [
            {
                "path": str(info.path),
                "num_samples": info.num_samples,
                "offset": info.offset,
                "split": info.split,
            }
            for info in clip_infos
        ],
    }

    meta_path = output_dir / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    summary = [
        f"Wrote {len([c for c in clip_infos if c.split == 'train'])} clips to {train_bin}",
        f"Wrote {len([c for c in clip_infos if c.split == 'val'])} clips to {val_bin}",
        f"Metadata saved to {meta_path}",
    ]
    sys.stdout.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
