# Digit trajectory sweep script guide

Run all commands below from the repository root. The sweep is a pipeline: the
sweep driver chooses configurations, the demo runs one configuration, the
dataset builder changes token availability between phases, the trainer writes
checkpoints, the exporter converts checkpoints to browser data, and the
manifest builder publishes each completed run to the selector.

## Scripts in the sweep pipeline

### `demos/digits_3d_trajectory_sweep.sh`

The top-level sweep driver. It builds the Cartesian product of embedding
dimensions, vocabulary sizes, radius modes, weight tying, optimizers, Adam
weight decays, affected-token counts, and transition schedules. For each
configuration it:

1. Creates a unique run name and output paths.
2. Passes the configuration through environment variables.
3. Runs `demos/digits_3d_trajectory_demo.sh`.
4. Runs `analysis/update_3d_sweep_manifest.py` after the trajectory completes.

Start the default sweep with:

```bash
bash demos/digits_3d_trajectory_sweep.sh
```

The training checkpoints go under `out/digits_3d_sweep/`. Browser-ready JSON
runs go under `report/threejs/digits-3d/runs/`.

### `demos/digits_3d_trajectory_demo.sh`

The single-run orchestrator. The sweep invokes this script once for each
configuration. It translates environment variables into dataset and training
arguments, calculates transition boundaries, and implements the three schedule
types:

- `drop`: train with affected tokens, remove them once, then resume.
- `add`: train without affected tokens, add them once, then resume.
- `duty_cycle`: repeatedly alternate included and excluded phases.

It saves a resumable checkpoint at every phase boundary and invokes the
trajectory exporter when training finishes. It can also be run independently:

```bash
bash demos/digits_3d_trajectory_demo.sh
```

### `data/digits_3d/prepare.py`

Builds `data/digits_3d/train.bin`, `val.bin`, and `meta.pkl`. Trained symbols
are written to the train and validation splits. Held-out letters and affected
symbols remain in the vocabulary even when they are absent from those splits,
which makes it possible to compare their embedding motion. The single-run
orchestrator calls this script again at each transition to change which symbols
are active without changing the vocabulary layout.

### `train.py`

The repository's main training entry point. The single-run orchestrator calls
it once per schedule phase with the small digit-model architecture, selected
optimizer, embedding constraints, checkpoint interval, and phase-ending
iteration. Later phases use `--init_from resume` to continue from the preceding
phase's `ckpt.pt`. Major numbered checkpoints provide the trajectory snapshots.

### `analysis/export_3d_token_trajectories.py`

Reads the numbered checkpoints and `data/digits_3d/meta.pkl`, extracts token
embedding coordinates and loss metrics, and writes one trajectory JSON file.
Native 2D coordinates are placed in the XY plane, native 3D coordinates are
kept unchanged, and higher-dimensional coordinates are projected with one PCA
basis fitted across all tokens and checkpoint frames. It also records schedule,
optimizer, fixed-radius, and weight-tying metadata for the browser.

### `analysis/update_3d_sweep_manifest.py`

Scans completed `dim-*.json` trajectories in
`report/threejs/digits-3d/runs/` and atomically rebuilds
`runs/manifest.json`. The selector uses this compact manifest to populate its
filters and run list. Updating it after each completed run lets the results be
browsed while the remainder of the sweep is still training.

## Browser files that consume the results

These are not executed by the shell sweep, but they are the final consumers of
its generated files:

- `report/threejs/digits-3d/index.html` loads `runs/manifest.json`, provides the
  sweep filters and run list, stores the filters and selected run in the URL,
  and embeds the selected trajectory viewer.
- `report/threejs/digits-3d/viewer.html` loads a selected trajectory JSON and
  renders token positions, trails, schedule state, and loss history with
  Three.js.

Serve the repository after or during a sweep:

```bash
python3 -m http.server 8000
```

Then open
`http://localhost:8000/report/threejs/digits-3d/index.html`.

## Optional packaging scripts

`demos/package_digits_3d_github_pages.sh` is not called by the sweep. Run it
after trajectories exist to invoke `analysis/package_3d_trajectory_site.py`,
which rebuilds the manifest and copies the selector, viewer, and completed run
JSON files into a standalone static-site directory.
