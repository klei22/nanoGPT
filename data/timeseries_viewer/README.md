# Time-series Prediction Viewer

`generate_timeseries_comparison.py` builds a small holdout comparison bundle for
integer multicontext CSV time series:

1. Split a prepared full CSV into an inference prompt and a tail holdout set.
2. Run `sample.py` for every requested random seed at both `top_k=1` and
   `top_k=5` (configurable).
3. Write an HTML page that overlays the prompt tail, ground-truth holdout, and
   every sampled forecast for each column.

The generated HTML is self-contained and uses browser canvas rendering, so it
has no Plotly or JavaScript package dependency.

## Dukascopy example

After training `demos/dukascopy_fx_m1_csv_mc_int_demo.sh`, run:

```bash
python3 data/timeseries_viewer/generate_timeseries_comparison.py \
  --input_csv data/dukascopy_fx_m1/input.csv \
  --manifest data/dukascopy_fx_m1/manifest.json \
  --checkpoint_dir out/dukascopy_fx_m1 \
  --work_dir out/dukascopy_fx_m1/timeseries_viewer \
  --holdout_rows 128 \
  --prompt_rows 512 \
  --seeds 1337 1338 1339 \
  --top_k 1 5
```

Open:

```text
out/dukascopy_fx_m1/timeseries_viewer/timeseries_prediction_vs_truth.html
```

The holdout rows are excluded from the prompt sent to `sample.py`, so they are
only used as ground truth for the graph.

For a quick viewer smoke test without a checkpoint, add `--skip_sampling`; this
writes the prompt, holdout CSV, and HTML with just the ground-truth series.
