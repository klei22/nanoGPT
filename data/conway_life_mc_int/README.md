# Conway Life Multicontext Dataset

`data/conway_life_mc_int` creates small integer-video curricula in the same
CSV/multicontext shape as `data/csv_mc_int`: every column becomes its own
regular nanoGPT dataset, and rendered cells are named `p0`, `p1`, ... so the
shared roomba grayscale viewer can play the frames.

The generator uses toroidal cellular automata with several Life-like rules:
standard Conway Life, HighLife, Seeds, and Day & Night. Episodes start from a
mix of random fields and named motifs such as gliders, blinkers, beacons,
toads, R-pentominoes, and pulsars. A small mutation probability injects random
cell flips after each update, which makes the pre-curriculum less perfectly
periodic while preserving local rule structure.

## Build the default sample dataset

```bash
data/conway_life_mc_int/get_dataset.sh
```

The default is a lightweight 8x8, 4-episode, 16-step sample that writes:

- `data/conway_life_mc_int/input.csv` for viewer/debug use.
- `data/conway_life_mc_int/manifest.json`.
- One folder per context/column (`timestamp`, `episode`, `rule_id`, `p0`, ...),
  each containing `train.bin`, `val.bin`, and `meta.pkl`.

Scale it up for real pre-curriculum runs with options like:

```bash
data/conway_life_mc_int/get_dataset.sh \
  --width 24 --height 24 \
  --episodes 64 --steps 128 \
  --mutation_chance 0.01 \
  --output_root conway_life_24x24
```

## View it

Open `data/roomba/roomba_grayscale_viewer.html` in a browser and choose
`data/conway_life_mc_int/input.csv`.

The viewer auto-detects `width` and `height` columns, discovers all `p*` pixel
columns, and renders the Life frame as grayscale. The same viewer still supports
Roomba pose data when pose columns are present.

## Train with multicontext

Use the ordered dataset list from `manifest.json` as the multicontext dataset
list for the existing training scripts. The metadata contexts (`rule_id`,
`pattern_id`, `alive_count`, `born_count`, `died_count`, etc.) let the model see
which dynamics produced each pixel frame instead of learning only pixels.

## Other grid/video pre-curricula to try

- **Elementary cellular automata strips**: one-dimensional Wolfram rules stacked
over time; great for controllable local dependencies and rule-conditioning.
- **Langton's ant / multi-ant worlds**: sparse agents leave pixel trails, which
adds long-horizon memory and reversible-looking trajectories.
- **Reaction-diffusion / Gray-Scott fields**: quantized grayscale chemistry-like
patterns with continuous-looking texture and slow morphology changes.
- **Falling sand / fluid toy worlds**: local conservation, gravity, obstacles,
and stochastic particle interactions in the same `p*` image format.
- **Maze growth and solver traces**: condition on algorithm id, then predict the
next occupancy/visited/frontier grid.
- **Synthetic sprite physics**: bouncing balls, occlusions, collision metadata,
and RGB or grayscale `p*` channels for very small video dynamics.

## One-command demo

Run the full flow and open the viewer with the generated CSV preloaded. The
training/sampling steps use `train.py` and `sample.py`, so install the repo's
Python dependencies first, for example `pip install -r requirements_cpu.txt`.

```bash
data/conway_life_mc_int/demo.sh
```

The demo runs `get_dataset.sh`, validates the generated manifest, trains a tiny
multicontext model, uses the first rows of the validation split as the sampling
prompt, samples 100 new frames by default, appends the sampled rows to the
validation CSV, starts a local HTTP server, and prints both the regular viewer
URL and the optional side-by-side comparison viewer URL. The regular viewer URL
marks start-token frames, validation ground truth, and sampled continuation
frames in one timeline. The comparison viewer steps the validation ground-truth
continuation and sampled continuation side by side, and can export a selected
range of the prompt+sample inference timeline as a WebM video.
Pass generation options after `--`, for example:

```bash
data/conway_life_mc_int/demo.sh -- --width 16 --height 16 --episodes 8 --steps 32
```

For headless checks, avoid opening a browser and stop the server automatically:

```bash
data/conway_life_mc_int/demo.sh --no-open --serve-seconds 2 --train-iters 2 --sample-frames 4
```
