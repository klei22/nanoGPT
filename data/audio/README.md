# Audio analysis sandbox (`data/audio`)

This folder mirrors the Whisper-mel CSV flow from `data/template` and adds a toy path into the integer multicontext CSV demo.

## Symlinks included

- `prepare.py -> ../template/prepare.py`
- `utils -> ../template/utils`

## Main scripts

- `generate_sine_wav.py`: create a dummy sine-wave WAV.
- `run_sine_audio_roundtrip.sh`: generate + (optionally) play + encode to mel CSV + decode to WAV + (optionally) play + emit CSV for integer demo.
- `mel_csv_to_numint_csv.py`: pick mel-bin columns and write a headered CSV for `prepare_csv_int_multicontext.py`.

## End-to-end quickstart

From repository root:

```bash
bash data/audio/run_sine_audio_roundtrip.sh
```

Outputs:
- `data/audio/dummy_sine.wav`
- `data/audio/dummy_sine_mel.csv`
- `data/audio/dummy_sine_recovered.wav`
- `data/audio/audio_num_int_input.csv`

Then run training+sampling demo:

```bash
bash demos/num_int_csv_audio.sh \
  data/audio/dummy_sine_mel.csv \
  data/audio/audio_num_int_input.csv \
  csv_num_mc_int_audio
```

This writes:
- quantized datasets under `data/csv_num_mc_int_audio/`
- model artifacts under `out/numerical_mc_csv_int_audio/`
- Plotly graph at `out/numerical_mc_csv_int_audio/num_int_csv_audio_samples.html`

## Notes

- Playback attempts `ffplay`, then `aplay`, then `play` if installed.
- Bin selection defaults to `10,30,60`; adjust via `--bins` in `mel_csv_to_numint_csv.py`.
