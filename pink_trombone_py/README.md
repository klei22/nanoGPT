# pink_trombone_py

This directory provides a minimal Python port of the [Pink Trombone](https://github.com/khuong291/pink-trombone) speech synthesizer.

The module exposes a small demo that converts text to IPA using `espeak-ng` and then synthesizes speech using the pink-trombone algorithm.

## Requirements

- Python 3.11+
- `numpy`, `soundfile`, and `espeakng` Python packages
- `espeak-ng` command line tool (for IPA conversion)

Install the Python dependencies with

```bash
pip install numpy soundfile espeakng
```

and install the `espeak-ng` binary using your package manager, e.g.

```bash
sudo apt-get install espeak-ng
```

## Usage

Run the demo module with a short text string and provide an output path for the generated WAV file:

```bash
python -m pink_trombone_py.demo "a" --out demo.wav
```

On success the script prints

```
Wrote demo.wav
```

and you will find the resulting audio in `demo.wav`.
