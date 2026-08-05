# Mel integer multicontext

`data/mel_mc_int` bridges the self-describing mel CSVs from
`data/mel_spectrogram` with the integer CSV multicontext pattern used by
`data/csv_mc_int`. Each mel band (`mel_000_q`, `mel_001_q`, ...) becomes one
regular nanoGPT multicontext dataset with token IDs equal to the quantized mel
state.

## Defaults

`run.sh` intentionally mirrors the high-fidelity settings in
`data/mel_spectrogram/run.sh`:

- preset `max`
- sample rate `48000`
- `fmin=10`, `fmax=20000`
- `384` mel columns per timestep
- `64` states per column
- `15 ms` timestep and `60 ms` window
- `n_fft=8192`, `top_db=96`
- `reference-mode=file_percentile`

## Build a dataset

```bash
bash data/mel_mc_int/run.sh path/to/audio.wav mel_mc_int
```

This writes the intermediate mel CSV/PNG under `data/mel_mc_int/mel_out/` and
the multicontext dataset under `data/mel_mc_int/` by default. The manifest lists
all mel-band dataset names for training or inference:

```bash
python3 - <<'PY'
import json
m = json.load(open('data/mel_mc_int/manifest.json'))
print(' '.join(m['multicontext_datasets']))
PY
```

## Inference and continuation demo

After training a multicontext model on the generated manifest, run:

```bash
bash data/mel_mc_int/demo_infer.sh out/your_run path/to/audio.wav 4.5 200
```

The demo encodes the selected audio, cuts the prompt at `4.5` seconds, asks
`sample.py` to continue all mel-band contexts, wraps the generated CSV with the
reference mel metadata, reconstructs `generated.wav`, and writes a small
`index.html` viewer. The viewer includes a browser-side file picker and cutoff
field for auditioning/selecting the next source file before re-running the demo
command.
