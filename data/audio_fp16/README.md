# Audio FP16 dataset builder

This folder contains a lightweight audio pre-processing flow that converts arbitrary audio files into float16 waveforms and packs them into the `train.bin`/`val.bin` files expected by the training pipeline. Each audio clip is also saved as its own `.npy` file so every datapoint can be inspected or re-used independently.

## Requirements

- Python 3.9+
- `numpy` and `soundfile` (`pip install numpy soundfile`)

`soundfile` relies on libsndfile for decoding; most common formats such as WAV/FLAC/OGG work out of the box. If your build of libsndfile has MP3 support enabled, MP3 inputs are also accepted.

## Usage

```bash
# From the repository root
python data/audio_fp16/prepare_audio_fp16.py \
  --inputs /path/to/audio1.wav /path/to/audio2.flac \
  --output_dir data/audio_fp16 \
  --target_sample_rate 16000 \
  --val_fraction 0.1
```

What the script does:

1. Loads each audio file, converts to mono, optionally resamples to `--target_sample_rate`, and normalizes to `[-1, 1]`.
2. Saves the float16 waveform for every clip into `processed_clips/<stem>.npy`.
3. Views the float16 values as unsigned 16-bit integers so they can be stored in `train.bin`/`val.bin` without losing information.
4. Writes a `meta.pkl` that records the offsets and lengths of each clip so downstream code can rehydrate the float16 values on demand.

The resulting `meta.pkl` advertises a `tokenizer` named `"audio_fp16"` and sets `value_dtype=float16`/`storage_dtype=uint16`, allowing the training loop to reinterpret batch slices back into fp16 values for numerical multi-context setups.
