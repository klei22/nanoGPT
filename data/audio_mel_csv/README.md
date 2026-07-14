# Audio mel CSV multicontext extension

This directory extends the existing Whisper-style mel CSV direction into an end-to-end workflow similar to the Dukascopy multicontext demo:

1. Decompose one audio file or a folder of audio files (`wav`, `mp3`, `flac`, `ogg`, or `m4a`) into Whisper-style normalized mel spectrogram frames.
2. Quantize each mel channel to integer CSV columns so `data/csv_mc_int` can create one regular multicontext dataset per channel.
3. Train with `train.py --training_mode multicontext` exactly like the Dukascopy flow.
4. Sample mel continuations, reconstruct approximate WAV files, and render a Plotly HTML page with playable audio.

The generated viewer intentionally reconstructs and embeds audio for **prompt/start tokens + continuation** for both model samples and ground truth.

## Prepare a dataset

```bash
python3 data/audio_mel_csv/prepare_audio_mel_csv.py sample.wav \
  --output_root audio_mel_csv \
  --work_dir data/audio_mel_csv/work
```

You can also pass a folder; supported audio files are processed in sorted order and concatenated at the mel-frame level.

Outputs:

- `data/audio_mel_csv/work/mel_float.csv`: normalized mel frames.
- `data/audio_mel_csv/work/mel_int.csv`: integer CSV with headers `mel_000` ... `mel_079`.
- `data/audio_mel_csv/manifest.json`: multicontext dataset manifest and mel/quantization metadata.
- `data/audio_mel_csv/mel_*/train.bin`, `val.bin`, and `meta.pkl`: one context per mel channel.

## Recompose a quantized mel CSV into audio

```bash
python3 data/audio_mel_csv/mel_int_csv_to_wav.py \
  data/audio_mel_csv/work/mel_int.csv \
  --output reconstructed.wav
```

This is an approximate inverse using `data/template/mel_csv_to_wav.py` and Griffin-Lim.

## Generate the audio viewer

After training a checkpoint, run:

```bash
python3 data/audio_mel_csv/generate_audio_mel_comparison.py \
  --input_csv data/audio_mel_csv/work/mel_int.csv \
  --manifest data/audio_mel_csv/manifest.json \
  --checkpoint_dir out/audio_mel_csv
```

Open `out/audio_mel_csv/audio_viewer/audio_prediction_vs_truth.html` to play sample and ground-truth audio and inspect Plotly heatmaps.

## End-to-end demo with generated audio

The demo accepts either an audio file or a folder of audio files. If no argument
is provided, or if the target folder does not contain supported audio files, the
script uses `sox` to synthesize a small folder of demo WAV files before running
the same prepare → train → sample → viewer pipeline.

```bash
bash demos/audio_mel_csv_mc_int_demo.sh [optional-audio-file-or-folder]
```
