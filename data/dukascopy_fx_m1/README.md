# Dukascopy FX M1 CSV Multicontext Data

This folder contains a small Dukascopy FX minute-candle workflow for the regular
integer CSV multicontext pipeline used by `data/csv_mc_int`.

## 1. Download raw candles

```bash
python3 data/dukascopy_fx_m1/download_dukascopy_fx_m1.py \
  --start 2025-01-02 --end 2025-01-03 \
  --universe majors \
  --out data/dukascopy_fx_m1/raw \
  --side BID \
  --max-workers 4 --rps 2
```

The downloader writes one gzipped CSV per instrument/day, for example
`data/dukascopy_fx_m1/raw/eurusd/2025-01-02_bid_m1.csv.gz`, with columns:

```text
timestamp_utc,open,high,low,close,volume
```

Dukascopy's historical data URLs use zero-based months; the downloader handles
that internally.

## 2. Build integer multicontext datasets

```bash
data/dukascopy_fx_m1/get_dataset.sh data/dukascopy_fx_m1/raw/eurusd
```

`get_dataset.sh` first converts the raw floating-point candle CSV into
`data/dukascopy_fx_m1/input.csv` with compact integer columns:

- `minute_mod_10` in `[0, 9]`
- `minute_of_hour` in `[0, 59]`
- `minute_of_day` in `[0, 1439]`
- `minute_of_week` in `[0, 10079]`
- `minute_of_year` in `[0, 527039]` (leap-year-safe minute index)
- `open_delta_state`, `high_delta_state`, `low_delta_state`,
  `close_delta_state`, and `volume_delta_state` as compact derivative states

For each OHLCV tick column, the converter computes the present tick value minus
the previous tick value, derives default 5th/95th percentile saturation
thresholds, clips outliers to those thresholds, and buckets the clipped
derivative into `DUKASCOPY_DELTA_STATES` discrete states. The raw absolute tick
values are not used as model tokens, which keeps vocabularies much smaller than
full price/volume ranges.

The converter also writes `stats.json` plus dependency-free PNG histograms for
both the original tick values and derivative values under `data/dukascopy_fx_m1/stats/`.

Then it calls `data/csv_mc_int/get_dataset.sh`, producing per-column datasets
under `data/dukascopy_fx_m1/` and a `manifest.json` containing the ordered
`multicontext_datasets` list consumed by the demo script.

Environment overrides:

- `DUKASCOPY_MC_OUTPUT_ROOT` (default `dukascopy_fx_m1`)
- `DUKASCOPY_PRICE_SCALE` (default `100000`)
- `DUKASCOPY_VOLUME_SCALE` (default `1000`)
- `DUKASCOPY_DELTA_STATES` (default `257`)
- `DUKASCOPY_DELTA_LOWER_PERCENTILE` / `DUKASCOPY_DELTA_UPPER_PERCENTILE` (default `5` / `95`)
- `DUKASCOPY_DELTA_THRESHOLDS`, space-separated overrides like `open:-20:20 volume:-5000:5000`
- `DUKASCOPY_LOG_DELTA=1` to apply signed `log1p` before derivative bucketing
- `DUKASCOPY_STATS_DIR` (default `data/dukascopy_fx_m1/stats`)

## 3. Train/sample demo

```bash
demos/dukascopy_fx_m1_csv_mc_int_demo.sh
```

The demo mirrors `demos/csv_mc_int_demo.sh`: it reads the generated manifest,
trains regular `--training_mode multicontext`, and samples CSV continuations
using `sample.py --multicontext_csv_input`. If the default raw input
`data/dukascopy_fx_m1/raw/eurusd` does not exist yet, the demo first downloads
one day of BID data for the built-in majors universe, then builds the EUR/USD
dataset from the downloaded CSVs.

Demo download overrides:

- `DUKASCOPY_DEMO_START` / `DUKASCOPY_DEMO_END` (default `2025-01-02` / `2025-01-03`)
- `DUKASCOPY_DEMO_SIDE` (default `BID`)
- `DUKASCOPY_DEMO_UNIVERSE` (default `majors`)
- `DUKASCOPY_DEMO_RAW_OUT` (default `data/dukascopy_fx_m1/raw`)
- `DUKASCOPY_DEMO_MAX_WORKERS` (default `4`)
- `DUKASCOPY_DEMO_RPS` (default `2`)

## Source and license

Raw candle data is downloaded from Dukascopy's public historical data feed:
<https://datafeed.dukascopy.com/datafeed>. Review Dukascopy's terms before
redistributing downloaded data.
