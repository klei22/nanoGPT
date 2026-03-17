# test_gemma_en_ko_heatmap.py
"""Test Gemma 270M tokenization on English-Korean translation pairs with heatmaps.

Fetches EN-KO translation pairs from the OPUS/Helsinki-NLP dataset via
HuggingFace, tokenizes both sides with the Gemma tokenizer, and produces:

  1. **Terminal heatmap** – ANSI-coloured token grids showing token density,
     byte-per-token ratio, and character-per-token ratio.
  2. **Interactive HTML heatmap** – a self-contained page with hover tooltips,
     side-by-side EN/KO comparison, and switchable colour modes.

Requirements: transformers, datasets, rich (optional for terminal)

Usage
-----
```bash
# quick test – 5 sentence pairs, terminal only
python test_gemma_en_ko_heatmap.py

# full run – 20 pairs, both outputs
python test_gemma_en_ko_heatmap.py \
    --num_pairs 20 \
    --html en_ko_heatmap.html \
    --output_file en_ko_terminal.txt
```
"""
from __future__ import annotations

import argparse
import colorsys
import html as html_lib
import io
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# colour helpers (shared with gemma_tokenization_highlighter.py)
# ---------------------------------------------------------------------------

def _hue_hex(hue: float, s: float = 0.7, l: float = 0.5) -> str:
    r, g, b = colorsys.hls_to_rgb(hue, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _rg_gradient(norm: float) -> str:
    """Red (0) -> Yellow (0.5) -> Green (1)."""
    norm = max(0.0, min(1.0, norm))
    r = int((1 - norm) * 255)
    g = int(norm * 255)
    return f"#{r:02x}{g:02x}00"


def _bg_style(hex_col: str) -> str:
    """Determine black/white foreground for readability."""
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "#000" if lum > 140 else "#fff"
    return f"background:{hex_col};color:{fg}"


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def tokenize_text(text: str, tokenizer) -> List[Tuple[int, str, str]]:
    """Return [(token_id, token_string, original_segment), ...]."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = []
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        tok_str = tokenizer.convert_ids_to_tokens(tid)
        spans.append((tid, tok_str, text[s:e]))
    return spans


def compute_metrics(spans: List[Tuple[int, str, str]]) -> Dict[str, float]:
    """Aggregate tokenization metrics."""
    if not spans:
        return {"num_tokens": 0, "total_chars": 0, "total_bytes": 0,
                "chars_per_token": 0, "bytes_per_token": 0}
    total_chars = sum(len(o) for _, _, o in spans)
    total_bytes = sum(len(o.encode("utf-8")) for _, _, o in spans)
    n = len(spans)
    return {
        "num_tokens": n,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "chars_per_token": total_chars / n,
        "bytes_per_token": total_bytes / n,
    }


# ---------------------------------------------------------------------------
# fetch EN-KO pairs
# ---------------------------------------------------------------------------

def fetch_en_ko_pairs(num_pairs: int, dataset_name: str, dataset_config: str) -> List[Tuple[str, str]]:
    """Load EN-KO translation pairs from HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name} ({dataset_config})...")
    ds = load_dataset(dataset_name, dataset_config, split=f"train[:{num_pairs}]")
    pairs = []
    for row in ds:
        tr = row["translation"]
        pairs.append((tr["en"], tr["ko"]))
    return pairs


# ---------------------------------------------------------------------------
# terminal rendering
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def render_terminal_heatmap(
    pairs: List[Tuple[str, str]],
    en_all: List[List[Tuple[int, str, str]]],
    ko_all: List[List[Tuple[int, str, str]]],
) -> str:
    """Render side-by-side EN/KO heatmaps to terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        lines = []
        for i, (en, ko) in enumerate(pairs):
            lines.append(f"--- Pair {i+1} ---")
            lines.append(f"EN: {en}")
            lines.append(f"KO: {ko}")
            lines.append(f"EN tokens: {len(en_all[i])}  KO tokens: {len(ko_all[i])}")
            lines.append("")
        return "\n".join(lines)

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor", width=160)

    # Summary table
    summary = Table(title="EN-KO Tokenization Comparison (Gemma 270M)", box=None, pad_edge=False)
    summary.add_column("#", justify="right", style="dim")
    summary.add_column("EN tokens", justify="right")
    summary.add_column("KO tokens", justify="right")
    summary.add_column("Ratio KO/EN", justify="right")
    summary.add_column("EN bytes/tok", justify="right")
    summary.add_column("KO bytes/tok", justify="right")

    for i, (en_spans, ko_spans) in enumerate(zip(en_all, ko_all)):
        en_m = compute_metrics(en_spans)
        ko_m = compute_metrics(ko_spans)
        ratio = ko_m["num_tokens"] / max(en_m["num_tokens"], 1)
        ratio_col = _rg_gradient(1 - min(ratio / 3, 1))  # green when close to 1

        summary.add_row(
            str(i + 1),
            str(en_m["num_tokens"]),
            str(ko_m["num_tokens"]),
            Text(f"{ratio:.2f}", style=f"bold {ratio_col}"),
            f"{en_m['bytes_per_token']:.1f}",
            f"{ko_m['bytes_per_token']:.1f}",
        )

    console.print(summary)
    console.print()

    # Per-pair token heatmaps
    for i, ((en_text, ko_text), en_spans, ko_spans) in enumerate(
        zip(pairs, en_all, ko_all)
    ):
        console.print(f"[bold cyan]--- Pair {i+1} ---[/bold cyan]")

        # EN tokens coloured by byte length
        en_line = Text()
        byte_lengths_en = [len(o.encode("utf-8")) for _, _, o in en_spans]
        lo_en, hi_en = (min(byte_lengths_en), max(byte_lengths_en)) if byte_lengths_en else (0, 1)
        for (tid, tok, orig), bl in zip(en_spans, byte_lengths_en):
            norm = (bl - lo_en) / (hi_en - lo_en + 1e-9)
            col = _rg_gradient(1 - norm)  # short bytes = green, long = red
            en_line.append(f"|{_escape(orig)}", style=f"bold {col}")
        en_line.append("|", style="dim")

        # KO tokens coloured by byte length
        ko_line = Text()
        byte_lengths_ko = [len(o.encode("utf-8")) for _, _, o in ko_spans]
        lo_ko, hi_ko = (min(byte_lengths_ko), max(byte_lengths_ko)) if byte_lengths_ko else (0, 1)
        for (tid, tok, orig), bl in zip(ko_spans, byte_lengths_ko):
            norm = (bl - lo_ko) / (hi_ko - lo_ko + 1e-9)
            col = _rg_gradient(1 - norm)
            ko_line.append(f"|{_escape(orig)}", style=f"bold {col}")
        ko_line.append("|", style="dim")

        console.print(Text("EN: ", style="bold"), end="")
        console.print(en_line)
        console.print(Text("KO: ", style="bold"), end="")
        console.print(ko_line)
        console.print()

    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _js_str(s: str) -> str:
    return (
        '"'
        + s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("'", "\\'")
        + '"'
    )


def render_html_heatmap(
    pairs: List[Tuple[str, str]],
    en_all: List[List[Tuple[int, str, str]]],
    ko_all: List[List[Tuple[int, str, str]]],
    title: str = "Gemma 270M – EN↔KO Tokenization Heatmap",
) -> str:
    """Build interactive HTML with side-by-side heatmaps."""

    # Build JS data
    pair_data_js = []
    for i, ((en_text, ko_text), en_spans, ko_spans) in enumerate(
        zip(pairs, en_all, ko_all)
    ):
        en_toks = ",".join(
            f'{{id:{tid},tok:{_js_str(tok)},orig:{_js_str(orig)},'
            f'chars:{len(orig)},bytes:{len(orig.encode("utf-8"))}}}'
            for tid, tok, orig in en_spans
        )
        ko_toks = ",".join(
            f'{{id:{tid},tok:{_js_str(tok)},orig:{_js_str(orig)},'
            f'chars:{len(orig)},bytes:{len(orig.encode("utf-8"))}}}'
            for tid, tok, orig in ko_spans
        )
        pair_data_js.append(
            f'{{en_text:{_js_str(en_text)},ko_text:{_js_str(ko_text)},'
            f'en_tokens:[{en_toks}],ko_tokens:[{ko_toks}]}}'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(title)}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0d1117; color:#e6edf3; padding:20px; }}
  h1 {{ font-size:1.5em; margin-bottom:8px; color:#58a6ff; }}
  .subtitle {{ color:#8b949e; margin-bottom:20px; font-size:0.9em; }}
  .controls {{ margin-bottom:16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
  .controls label {{ font-size:0.85em; color:#8b949e; }}
  .controls select {{ background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:4px 10px; }}
  .pair {{ background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; margin-bottom:16px; }}
  .pair-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
  .pair-header h3 {{ font-size:1em; color:#58a6ff; }}
  .pair-stats {{ font-size:0.8em; color:#8b949e; }}
  .lang-row {{ margin-bottom:8px; }}
  .lang-label {{ display:inline-block; width:28px; font-weight:bold; color:#f0883e; font-size:0.85em; vertical-align:top; padding-top:4px; }}
  .token-row {{ display:inline; line-height:2.2; }}
  .tok {{
    display:inline-block; padding:2px 5px; margin:1px; border-radius:3px;
    cursor:pointer; font-family:'Fira Code','Consolas',monospace; font-size:0.9em;
    border:1px solid rgba(255,255,255,0.08); transition:transform 0.1s;
  }}
  .tok:hover {{ transform:scale(1.15); box-shadow:0 0 10px rgba(88,166,255,0.4); z-index:10; position:relative; }}
  #tooltip {{
    display:none; position:fixed; background:#1f2937; color:#fff;
    padding:10px 14px; border-radius:6px; font-size:0.82em; z-index:100;
    box-shadow:0 4px 16px rgba(0,0,0,0.6); max-width:350px;
    font-family:'Fira Code',monospace; line-height:1.7; border:1px solid #30363d;
  }}
  #tooltip .label {{ color:#58a6ff; }}
  .summary-table {{ width:100%; border-collapse:collapse; margin-bottom:20px; font-size:0.85em; }}
  .summary-table th {{ text-align:left; padding:8px; border-bottom:2px solid #30363d; color:#58a6ff; }}
  .summary-table td {{ padding:6px 8px; border-bottom:1px solid #21262d; }}
  .summary-table tr:hover {{ background:#21262d; }}
  .ratio-badge {{
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-weight:bold; font-size:0.85em;
  }}
</style>
</head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p class="subtitle">Hover over tokens for details. Korean text typically requires more tokens than English for the same semantic content.</p>

<div class="controls">
  <label>Colour by:
    <select id="colourMode">
      <option value="byte_length">Byte length (heatmap)</option>
      <option value="char_length">Char length</option>
      <option value="token_id">Token ID (hashed)</option>
    </select>
  </label>
</div>

<table class="summary-table" id="summary"></table>

<div id="pairs-container"></div>
<div id="tooltip"></div>

<script>
const data = [{",".join(pair_data_js)}];
const container = document.getElementById('pairs-container');
const tooltip = document.getElementById('tooltip');
const golden = 0.618033988749895;

function hueCol(id) {{
  const h = (id * golden) % 1.0;
  const s = 0.65, l = 0.5;
  const f = (n) => {{
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return Math.round((l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))) * 255);
  }};
  return '#' + [f(0),f(8),f(4)].map(x => x.toString(16).padStart(2,'0')).join('');
}}

function rgGrad(norm) {{
  const n = Math.max(0, Math.min(1, norm));
  const r = Math.round((1-n)*255), g = Math.round(n*255);
  return '#' + r.toString(16).padStart(2,'0') + g.toString(16).padStart(2,'0') + '00';
}}

function fgFor(bg) {{
  const r=parseInt(bg.slice(1,3),16), g=parseInt(bg.slice(3,5),16), b=parseInt(bg.slice(5,7),16);
  return (0.299*r+0.587*g+0.114*b) > 140 ? '#000' : '#fff';
}}

function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function renderSummary() {{
  const tbl = document.getElementById('summary');
  let html = '<tr><th>#</th><th>EN tokens</th><th>KO tokens</th><th>Ratio KO/EN</th><th>EN bytes/tok</th><th>KO bytes/tok</th></tr>';
  data.forEach((p, i) => {{
    const enN = p.en_tokens.length, koN = p.ko_tokens.length;
    const ratio = koN / Math.max(enN, 1);
    const enB = p.en_tokens.reduce((s,t)=>s+t.bytes,0) / Math.max(enN,1);
    const koB = p.ko_tokens.reduce((s,t)=>s+t.bytes,0) / Math.max(koN,1);
    const col = rgGrad(1 - Math.min(ratio/3, 1));
    html += `<tr><td>${{i+1}}</td><td>${{enN}}</td><td>${{koN}}</td>` +
      `<td><span class="ratio-badge" style="background:${{col}};color:${{fgFor(col)}}">${{ratio.toFixed(2)}}</span></td>` +
      `<td>${{enB.toFixed(1)}}</td><td>${{koB.toFixed(1)}}</td></tr>`;
  }});
  tbl.innerHTML = html;
}}

function colourTokens(tokens, mode) {{
  if (mode === 'token_id') return tokens.map(t => hueCol(t.id));
  const key = mode === 'char_length' ? 'chars' : 'bytes';
  const vals = tokens.map(t => t[key]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return vals.map(v => rgGrad(1 - (hi-lo < 1e-9 ? 0.5 : (v-lo)/(hi-lo))));
}}

function renderPairs() {{
  const mode = document.getElementById('colourMode').value;
  container.innerHTML = '';
  data.forEach((p, i) => {{
    const div = document.createElement('div');
    div.className = 'pair';

    const enCols = colourTokens(p.en_tokens, mode);
    const koCols = colourTokens(p.ko_tokens, mode);
    const enN = p.en_tokens.length, koN = p.ko_tokens.length;
    const ratio = (koN / Math.max(enN, 1)).toFixed(2);

    let html = `<div class="pair-header"><h3>Pair ${{i+1}}</h3>` +
      `<span class="pair-stats">EN: ${{enN}} tokens | KO: ${{koN}} tokens | ratio: ${{ratio}}</span></div>`;

    html += renderLangRow('EN', p.en_tokens, enCols);
    html += renderLangRow('KO', p.ko_tokens, koCols);
    div.innerHTML = html;
    container.appendChild(div);
  }});

  // Attach hover events
  container.querySelectorAll('.tok').forEach(el => {{
    el.addEventListener('mouseenter', (ev) => {{
      const d = el.dataset;
      tooltip.innerHTML =
        `<span class="label">Token ID:</span> ${{d.tid}}<br>` +
        `<span class="label">Token:</span> ${{escHtml(d.tok)}}<br>` +
        `<span class="label">Original:</span> ${{escHtml(d.orig)}}<br>` +
        `<span class="label">Chars:</span> ${{d.chars}}<br>` +
        `<span class="label">Bytes:</span> ${{d.bytes}}`;
      tooltip.style.display = 'block';
      pos(ev);
    }});
    el.addEventListener('mousemove', pos);
    el.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
  }});
}}

function renderLangRow(label, tokens, colours) {{
  let html = `<div class="lang-row"><span class="lang-label">${{label}}</span><span class="token-row">`;
  tokens.forEach((t, j) => {{
    let display = t.orig.replace(/\\n/g,'\\\\n').replace(/\\t/g,'\\\\t');
    if (display === ' ') display = '\\u2423';
    if (display === '') display = '\\u2205';
    const bg = colours[j], fg = fgFor(bg);
    html += `<span class="tok" style="background:${{bg}};color:${{fg}}" ` +
      `data-tid="${{t.id}}" data-tok="${{escHtml(t.tok)}}" data-orig="${{escHtml(t.orig)}}" ` +
      `data-chars="${{t.chars}}" data-bytes="${{t.bytes}}">${{escHtml(display)}}</span>`;
  }});
  html += '</span></div>';
  return html;
}}

function pos(ev) {{
  tooltip.style.left = (ev.clientX + 12) + 'px';
  tooltip.style.top = (ev.clientY + 12) + 'px';
}}

document.getElementById('colourMode').addEventListener('change', renderPairs);
renderSummary();
renderPairs();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test Gemma 270M tokenization on EN-KO translation pairs with heatmaps"
    )
    p.add_argument(
        "--model", type=str, default="google/gemma-3-270m",
        help="HuggingFace model for tokenizer",
    )
    p.add_argument(
        "--dataset", type=str, default="Helsinki-NLP/opus-100",
        help="HuggingFace translation dataset",
    )
    p.add_argument(
        "--dataset_config", type=str, default="en-ko",
        help="Dataset configuration / language pair",
    )
    p.add_argument("--num_pairs", type=int, default=5, help="Number of sentence pairs to process")
    p.add_argument("--html", type=str, default=None, help="Path for interactive HTML output")
    p.add_argument("--output_file", type=str, default=None, help="Path to save terminal ANSI output")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    pairs = fetch_en_ko_pairs(args.num_pairs, args.dataset, args.dataset_config)
    print(f"Loaded {len(pairs)} EN-KO pairs")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Tokenize all pairs
    en_all = [tokenize_text(en, tokenizer) for en, _ in pairs]
    ko_all = [tokenize_text(ko, tokenizer) for _, ko in pairs]

    # Terminal output
    ansi = render_terminal_heatmap(pairs, en_all, ko_all)
    print(ansi)

    if args.output_file:
        Path(args.output_file).write_text(ansi, "utf-8", errors="replace")
        print(f"Saved terminal output -> {args.output_file}")

    # HTML output
    if args.html:
        html_content = render_html_heatmap(pairs, en_all, ko_all)
        Path(args.html).write_text(html_content, "utf-8")
        print(f"Saved interactive HTML -> {args.html}")


if __name__ == "__main__":
    main()
