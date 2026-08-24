#!/usr/bin/env python3
"""Sample multicontext time-series forecasts and render prediction vs truth HTML."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SeriesRun:
    label: str
    top_k: int
    seed: int
    csv_path: Path


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
        rows = [row for row in reader if row]
    return header, rows


def write_csv(path: Path, header: list[str], rows: Iterable[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def numeric_rows(rows: list[list[str]]) -> list[list[float | None]]:
    parsed: list[list[float | None]] = []
    for row in rows:
        parsed_row: list[float | None] = []
        for cell in row:
            try:
                parsed_row.append(float(cell))
            except ValueError:
                parsed_row.append(None)
        parsed.append(parsed_row)
    return parsed


def load_manifest_datasets(path: Path) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    datasets = manifest.get("multicontext_datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"manifest has no multicontext_datasets: {path}")
    return [str(dataset) for dataset in datasets]


def run_sample(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    prompt_csv: Path,
    output_csv: Path,
    datasets: list[str],
    max_new_tokens: int,
    top_k: int,
    seed: int,
    device: str,
    dtype: str,
    compile_model: bool,
) -> None:
    cmd = [
        sys.executable,
        "sample.py",
        "--out_dir",
        str(checkpoint_dir),
        "--device",
        device,
        "--dtype",
        dtype,
        "--multicontext",
        "--multicontext_datasets",
        *datasets,
        "--multicontext_csv_input",
        str(prompt_csv),
        "--multicontext_csv_output_file",
        str(output_csv),
        "--no-multicontext_csv_output_include_prompt",
        "--max_new_tokens",
        str(max_new_tokens),
        "--top_k",
        str(top_k),
        "--seed",
        str(seed),
        "--num_samples",
        "1",
        "--no-print_model_info",
    ]
    if compile_model:
        cmd.append("--compile")
    else:
        cmd.append("--no-compile")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def write_viewer_html(
    *,
    output_html: Path,
    title: str,
    header: list[str],
    prompt_rows: list[list[str]],
    truth_rows: list[list[str]],
    runs: list[SeriesRun],
) -> None:
    samples = []
    for run in runs:
        if not run.csv_path.exists():
            continue
        sample_header, sample_rows = read_csv_rows(run.csv_path)
        samples.append(
            {
                "label": run.label,
                "top_k": run.top_k,
                "seed": run.seed,
                "header": sample_header,
                "rows": numeric_rows(sample_rows),
            }
        )

    payload = {
        "title": title,
        "header": header,
        "promptRows": numeric_rows(prompt_rows),
        "truthRows": numeric_rows(truth_rows),
        "samples": samples,
        "createdAt": datetime.now().isoformat(timespec="seconds"),
    }
    json_payload = json.dumps(payload).replace("</", "<\\/")
    escaped_title = html.escape(title)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escaped_title}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }}
    body {{ margin: 0; padding: 20px; background: #101114; color: #eee; }}
    h1 {{ margin-top: 0; font-size: 22px; }}
    .panel {{ background: #191b20; border: 1px solid #333842; border-radius: 10px; padding: 14px; margin-bottom: 16px; }}
    .controls {{ display: grid; grid-template-columns: minmax(220px, 1fr) minmax(320px, 3fr); gap: 16px; align-items: start; }}
    .global-toggles, .sample-toggles {{ display: flex; flex-wrap: wrap; gap: 10px 14px; }}
    label {{ display: inline-flex; gap: 6px; align-items: center; font-size: 13px; color: #cdd1d7; }}
    input {{ accent-color: #58a6ff; }}
    .chart {{ margin-bottom: 22px; }}
    .chart h2 {{ margin: 0 0 8px; font-size: 16px; }}
    .plot {{ width: 100%; height: 340px; border: 1px solid #343a46; border-radius: 8px; overflow: hidden; }}
    .swatch {{ width: 18px; height: 3px; display: inline-block; }}
    .meta {{ color: #aeb4bf; font-size: 13px; margin-bottom: 14px; }}
    .hint {{ color: #aeb4bf; font-size: 12px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <section class=\"panel controls\">
    <div>
      <strong>Global series</strong>
      <div class=\"global-toggles\">
        <label><input id=\"showPrompt\" type=\"checkbox\" checked /> prompt tail</label>
        <label><input id=\"showTruth\" type=\"checkbox\" checked /> ground truth holdout</label>
      </div>
      <div class=\"hint\">All columns are shown below at the same time as Plotly graphs.</div>
    </div>
    <div>
      <strong>Inference runs</strong>
      <div class=\"sample-toggles\" id=\"sampleToggles\"></div>
      <div class=\"hint\">Turn individual seeds/top-k runs on or off without regenerating the page.</div>
    </div>
  </section>
  <section class=\"panel\">
    <div class=\"meta\" id=\"meta\"></div>
    <div id=\"charts\"></div>
  </section>
<script id=\"payload\" type=\"application/json\">{json_payload}</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
const colors = ['#ffb000', '#00d1ff', '#f05cff', '#5cff92', '#ff6b6b', '#9b8cff', '#f6ff5c', '#58a6ff', '#ffa657', '#7ee787', '#d2a8ff', '#ff7b72', '#a5d6ff', '#f2cc60', '#56d364'];
document.getElementById('meta').textContent = `created ${{data.createdAt}} · columns ${{data.header.length}} · prompt rows ${{data.promptRows.length}} · holdout rows ${{data.truthRows.length}} · inference runs ${{data.samples.length}}`;
const sampleToggles = document.getElementById('sampleToggles');
data.samples.forEach((sample, idx) => {{
  const id = `sample_${{idx}}`;
  const label = document.createElement('label');
  label.innerHTML = `<input id="${{id}}" data-sample-idx="${{idx}}" type="checkbox" checked /> <span class="swatch" style="background:${{colors[idx % colors.length]}}"></span>${{sample.label}}`;
  sampleToggles.append(label);
}});
document.getElementById('showPrompt').addEventListener('change', render);
document.getElementById('showTruth').addEventListener('change', render);
sampleToggles.addEventListener('change', render);
function finiteOrNull(v) {{ return typeof v === 'number' && Number.isFinite(v) ? v : null; }}
function xRange(start, length) {{ return Array.from({{length}}, (_, idx) => start + idx); }}
function enabledSamplesForColumn(col) {{
  return data.samples
    .map((sample, idx) => ({{idx, label: sample.label, values: sample.rows.map(r => finiteOrNull(r[col]))}}))
    .filter(s => {{ const box = document.getElementById(`sample_${{s.idx}}`); return box && box.checked; }});
}}
function trace(name, x, y, color, width, dash='solid') {{
  return {{
    type: 'scatter',
    mode: 'lines',
    name,
    x,
    y,
    line: {{color, width, dash}},
    hovertemplate: 'row=%{{x}}<br>value=%{{y}}<extra>' + name + '</extra>'
  }};
}}
function renderColumn(col, charts) {{
  const holder = document.createElement('div'); holder.className = 'chart';
  const title = document.createElement('h2'); title.textContent = data.header[col];
  const plot = document.createElement('div'); plot.className = 'plot'; plot.id = `plot_${{col}}`;
  holder.append(title, plot); charts.append(holder);
  const prompt = data.promptRows.slice(Math.max(0, data.promptRows.length - 128)).map(r => finiteOrNull(r[col]));
  const truth = data.truthRows.map(r => finiteOrNull(r[col]));
  const promptStart = 0;
  const forecastStart = prompt.length;
  const traces = [];
  if (document.getElementById('showPrompt').checked) traces.push(trace('prompt tail', xRange(promptStart, prompt.length), prompt, '#8a96a8', 2));
  if (document.getElementById('showTruth').checked) traces.push(trace('ground truth holdout', xRange(forecastStart, truth.length), truth, '#ffffff', 3));
  enabledSamplesForColumn(col).forEach(s => traces.push(trace(s.label, xRange(forecastStart, s.values.length), s.values, colors[s.idx % colors.length], 2, 'dash')));
  const splitX = Math.max(forecastStart - 0.5, 0);
  const layout = {{
    paper_bgcolor: '#08090b',
    plot_bgcolor: '#08090b',
    font: {{color: '#d6deeb'}},
    margin: {{l: 58, r: 20, t: 12, b: 42}},
    hovermode: 'x unified',
    legend: {{orientation: 'h', x: 0, y: -0.22}},
    xaxis: {{title: 'row index relative to prompt tail', gridcolor: '#27303b', zerolinecolor: '#3a4350'}},
    yaxis: {{title: data.header[col], gridcolor: '#27303b', zerolinecolor: '#3a4350'}},
    shapes: [{{type: 'line', x0: splitX, x1: splitX, y0: 0, y1: 1, yref: 'paper', line: {{color: '#777', dash: 'dot', width: 1}}}}]
  }};
  Plotly.newPlot(plot, traces, layout, {{responsive: true, displaylogo: false}});
}}
function render() {{
  const charts = document.getElementById('charts'); charts.innerHTML = '';
  data.header.forEach((_, col) => renderColumn(col, charts));
}}
if (!window.Plotly) {{
  document.getElementById('charts').innerHTML = '<div class="panel">Plotly failed to load. Check network access to the Plotly CDN.</div>';
}} else {{
  render();
}}
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seeded top-k samples and render prediction-vs-ground-truth time-series HTML.")
    parser.add_argument("--input_csv", default="data/dukascopy_fx_m1/input.csv", help="Prepared integer CSV containing the full series.")
    parser.add_argument("--manifest", default="data/dukascopy_fx_m1/manifest.json", help="Dataset manifest with multicontext_datasets.")
    parser.add_argument("--checkpoint_dir", default="out/dukascopy_fx_m1", help="Checkpoint directory passed to sample.py --out_dir.")
    parser.add_argument("--work_dir", default="out/dukascopy_fx_m1/timeseries_viewer", help="Directory for prompt, truth, samples, and HTML.")
    parser.add_argument("--holdout_rows", type=int, default=128, help="Tail rows excluded from prompt and used as ground truth.")
    parser.add_argument("--prompt_rows", type=int, default=512, help="Rows immediately before holdout used as inference prompt; 0 uses all prior rows.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339, 1340, 1341], help="Random seeds to sample for each top-k value.")
    parser.add_argument("--top_k", nargs="+", type=int, default=[1, 5, 10], help="Top-k settings to run for every seed.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile", dest="compile_model", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_sampling", action="store_true", help="Only write prompt/truth and HTML; useful for checking the viewer without a checkpoint.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = repo_root / input_csv
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = repo_root / work_dir

    header, rows = read_csv_rows(input_csv)
    if args.holdout_rows <= 0:
        raise ValueError("--holdout_rows must be positive")
    if len(rows) <= args.holdout_rows:
        raise ValueError(f"Need more rows than holdout_rows={args.holdout_rows}; found {len(rows)}")
    prompt_source = rows[: -args.holdout_rows]
    prompt_rows = prompt_source if args.prompt_rows == 0 else prompt_source[-args.prompt_rows :]
    truth_rows = rows[-args.holdout_rows :]
    prompt_csv = work_dir / "prompt.csv"
    truth_csv = work_dir / "ground_truth_holdout.csv"
    write_csv(prompt_csv, header, prompt_rows)
    write_csv(truth_csv, header, truth_rows)

    datasets = load_manifest_datasets(manifest_path)
    runs: list[SeriesRun] = []
    samples_dir = work_dir / "samples"
    if not args.skip_sampling:
        for top_k in args.top_k:
            for seed in args.seeds:
                label = f"top_k={top_k} seed={seed}"
                output_csv = samples_dir / f"sample_topk{top_k}_seed{seed}.csv"
                run_sample(
                    repo_root=repo_root,
                    checkpoint_dir=checkpoint_dir,
                    prompt_csv=prompt_csv,
                    output_csv=output_csv,
                    datasets=datasets,
                    max_new_tokens=args.holdout_rows,
                    top_k=top_k,
                    seed=seed,
                    device=args.device,
                    dtype=args.dtype,
                    compile_model=args.compile_model,
                )
                runs.append(SeriesRun(label=label, top_k=top_k, seed=seed, csv_path=output_csv))
    else:
        for path in sorted(samples_dir.glob("sample_topk*_seed*.csv")):
            runs.append(SeriesRun(label=path.stem, top_k=-1, seed=-1, csv_path=path))

    output_html = work_dir / "timeseries_prediction_vs_truth.html"
    write_viewer_html(
        output_html=output_html,
        title="Time-series prediction vs ground truth",
        header=header,
        prompt_rows=prompt_rows,
        truth_rows=truth_rows,
        runs=runs,
    )
    print(f"Prompt CSV: {prompt_csv}")
    print(f"Ground truth CSV: {truth_csv}")
    print(f"Viewer HTML: {output_html}")


if __name__ == "__main__":
    main()
