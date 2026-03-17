# gemma_tokenization_highlighter.py
"""Visualise how the Gemma 270M tokenizer segments text.

Produces both a **terminal** (ANSI) view and an **interactive HTML** heatmap
that colour each token by its length in characters, byte-length, or token-id.

Works standalone with only `transformers` and (optionally) `rich` installed.

Usage
-----
```bash
# terminal output (default)
python gemma_tokenization_highlighter.py --text "Hello, world! 안녕하세요!"

# HTML output
python gemma_tokenization_highlighter.py \
    --text "Hello, world! 안녕하세요!" \
    --html tokenization.html

# read from file
python gemma_tokenization_highlighter.py --input_file sample.txt --html out.html
```
"""
from __future__ import annotations

import argparse
import colorsys
import html as html_lib
import io
import sys
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------

def _hue_to_rgb_hex(hue: float, saturation: float = 0.65, lightness: float = 0.50) -> str:
    """Convert HSL hue (0-1) to an #RRGGBB hex string."""
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _token_id_colour(token_id: int, vocab_size: int) -> str:
    """Deterministic colour based on token ID (golden-ratio hashing)."""
    golden = 0.618033988749895
    hue = (token_id * golden) % 1.0
    return _hue_to_rgb_hex(hue)


def _scalar_colour(value: float, lo: float, hi: float) -> str:
    """Red-to-green gradient for a scalar value between lo..hi."""
    if hi - lo < 1e-9:
        norm = 0.5
    else:
        norm = (value - lo) / (hi - lo)
    norm = max(0.0, min(1.0, norm))
    r = int((1 - norm) * 255)
    g = int(norm * 255)
    return f"#{r:02x}{g:02x}00"


# ---------------------------------------------------------------------------
# tokenize and build spans
# ---------------------------------------------------------------------------

def tokenize_to_spans(
    text: str,
    tokenizer,
) -> List[Tuple[int, str, str]]:
    """Return list of (token_id, token_string, original_text_segment)."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    spans = []
    for tid, (start, end) in zip(ids, offsets):
        tok_str = tokenizer.convert_ids_to_tokens(tid)
        original = text[start:end]
        spans.append((tid, tok_str, original))
    return spans


# ---------------------------------------------------------------------------
# colour mode implementations
# ---------------------------------------------------------------------------

COLOUR_MODES = ("token_id", "char_length", "byte_length")


def get_colours(
    spans: List[Tuple[int, str, str]],
    mode: str,
    vocab_size: int,
) -> List[str]:
    """Return one hex colour string per span."""
    if mode == "token_id":
        return [_token_id_colour(tid, vocab_size) for tid, _, _ in spans]

    if mode == "char_length":
        lengths = [len(orig) for _, _, orig in spans]
    else:  # byte_length
        lengths = [len(orig.encode("utf-8")) for _, _, orig in spans]

    lo, hi = (min(lengths), max(lengths)) if lengths else (0, 1)
    return [_scalar_colour(l, lo, hi) for l in lengths]


# ---------------------------------------------------------------------------
# terminal output (Rich)
# ---------------------------------------------------------------------------

def render_terminal(
    spans: List[Tuple[int, str, str]],
    colours: List[str],
    show_boundaries: bool = True,
) -> str:
    """Render coloured tokens to the terminal via Rich."""
    try:
        from rich.console import Console
        from rich.text import Text
    except ImportError:
        # Fallback: plain ANSI
        parts = []
        for (tid, tok_str, orig), col in zip(spans, colours):
            display = repr(orig)[1:-1] if show_boundaries else orig
            parts.append(f"\033[1m{display}\033[0m")
        return "".join(parts)

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor")
    text = Text()
    for (tid, tok_str, orig), col in zip(spans, colours):
        display = orig.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        if show_boundaries:
            display = f"|{display}"
        text.append(display, style=f"bold {col}")
    if show_boundaries:
        text.append("|", style="dim")
    console.print(text)

    # Also print summary table
    console.print()
    console.print(f"[cyan]Total tokens:[/cyan] {len(spans)}")
    console.print(f"[cyan]Total chars:[/cyan]  {sum(len(o) for _, _, o in spans)}")
    console.print(f"[cyan]Total bytes:[/cyan]  {sum(len(o.encode('utf-8')) for _, _, o in spans)}")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

def render_html(
    text: str,
    spans: List[Tuple[int, str, str]],
    colours: List[str],
    title: str = "Gemma 270M Tokenization",
) -> str:
    """Build a self-contained interactive HTML page."""
    token_data_js = []
    for i, ((tid, tok_str, orig), col) in enumerate(zip(spans, colours)):
        token_data_js.append(
            f'{{id:{tid},tok:{_js_str(tok_str)},orig:{_js_str(orig)},'
            f'chars:{len(orig)},bytes:{len(orig.encode("utf-8"))},col:"{col}"}}'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(title)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ margin-bottom: 10px; font-size: 1.4em; color: #e0e0ff; }}
  .stats {{ color: #aaa; margin-bottom: 16px; font-size: 0.9em; }}
  .controls {{ margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls label {{ font-size: 0.85em; color: #ccc; }}
  .controls select, .controls input {{ background: #2a2a4a; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }}
  #token-container {{ line-height: 2.2; padding: 16px; background: #16213e; border-radius: 8px; overflow-x: auto; }}
  .tok {{
    display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 3px;
    cursor: pointer; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.95em;
    transition: transform 0.1s, box-shadow 0.1s;
    border: 1px solid rgba(255,255,255,0.1);
  }}
  .tok:hover {{ transform: scale(1.1); box-shadow: 0 0 8px rgba(255,255,255,0.3); z-index: 10; position: relative; }}
  #tooltip {{
    display: none; position: fixed; background: #0f3460; color: #fff;
    padding: 10px 14px; border-radius: 6px; font-size: 0.85em; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 350px;
    font-family: 'Fira Code', monospace; line-height: 1.6;
  }}
  #tooltip .label {{ color: #7ec8e3; }}
</style>
</head>
<body>
<h1>{html_lib.escape(title)}</h1>
<div class="stats" id="stats"></div>
<div class="controls">
  <label>Colour by:
    <select id="colourMode">
      <option value="token_id">Token ID (hashed)</option>
      <option value="char_length">Char length</option>
      <option value="byte_length">Byte length</option>
    </select>
  </label>
  <label>Show boundaries:
    <input type="checkbox" id="showBounds" checked>
  </label>
</div>
<div id="token-container"></div>
<div id="tooltip"></div>

<script>
const tokens = [{",".join(token_data_js)}];
const container = document.getElementById('token-container');
const tooltip = document.getElementById('tooltip');
const stats = document.getElementById('stats');

stats.textContent = `Total tokens: ${{tokens.length}} | Total chars: ${{tokens.reduce((s,t)=>s+t.chars,0)}} | Total bytes: ${{tokens.reduce((s,t)=>s+t.bytes,0)}}`;

function hueToHex(h, s=0.65, l=0.5) {{
  const f = (n) => {{
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return Math.round((l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))) * 255);
  }};
  return '#' + [f(0),f(8),f(4)].map(x => x.toString(16).padStart(2,'0')).join('');
}}

function scalarCol(val, lo, hi) {{
  const n = hi-lo < 1e-9 ? 0.5 : Math.max(0, Math.min(1, (val-lo)/(hi-lo)));
  const r = Math.round((1-n)*255), g = Math.round(n*255);
  return '#' + r.toString(16).padStart(2,'0') + g.toString(16).padStart(2,'0') + '00';
}}

function recolour(mode) {{
  const spans = container.querySelectorAll('.tok');
  if (mode === 'token_id') {{
    spans.forEach((el, i) => {{ el.style.background = tokens[i].col; }});
  }} else {{
    const key = mode === 'char_length' ? 'chars' : 'bytes';
    const vals = tokens.map(t => t[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    spans.forEach((el, i) => {{ el.style.background = scalarCol(vals[i], lo, hi); }});
  }}
}}

function render() {{
  const showBounds = document.getElementById('showBounds').checked;
  container.innerHTML = '';
  tokens.forEach((t, i) => {{
    const el = document.createElement('span');
    el.className = 'tok';
    let display = t.orig.replace(/\\n/g, '\\\\n').replace(/\\t/g, '\\\\t').replace(/\\r/g, '\\\\r');
    if (display === ' ') display = '\\u2423';
    if (display === '') display = '\\u2205';
    el.textContent = display;
    el.style.background = t.col;
    el.style.color = '#fff';
    el.addEventListener('mouseenter', (ev) => {{
      tooltip.innerHTML =
        `<span class="label">Token ID:</span> ${{t.id}}<br>` +
        `<span class="label">Token:</span> ${{escHtml(t.tok)}}<br>` +
        `<span class="label">Original:</span> ${{escHtml(t.orig)}}<br>` +
        `<span class="label">Chars:</span> ${{t.chars}}<br>` +
        `<span class="label">Bytes:</span> ${{t.bytes}}`;
      tooltip.style.display = 'block';
      positionTooltip(ev);
    }});
    el.addEventListener('mousemove', positionTooltip);
    el.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
    container.appendChild(el);
  }});
  recolour(document.getElementById('colourMode').value);
}}

function positionTooltip(ev) {{
  tooltip.style.left = (ev.clientX + 12) + 'px';
  tooltip.style.top = (ev.clientY + 12) + 'px';
}}

function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

document.getElementById('colourMode').addEventListener('change', (e) => recolour(e.target.value));
document.getElementById('showBounds').addEventListener('change', render);
render();
</script>
</body>
</html>"""


def _js_str(s: str) -> str:
    """Escape a Python string for safe embedding in JS."""
    return (
        '"'
        + s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        + '"'
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise Gemma 270M tokenization with colour highlighting"
    )
    p.add_argument("--text", type=str, default=None, help="Text to tokenize")
    p.add_argument("--input_file", type=str, default=None, help="Read text from a file")
    p.add_argument(
        "--model", type=str, default="google/gemma-3-270m",
        help="HuggingFace model name for tokenizer",
    )
    p.add_argument(
        "--colour_mode", choices=COLOUR_MODES, default="token_id",
        help="How to assign colours to tokens",
    )
    p.add_argument("--html", type=str, default=None, help="Path to write interactive HTML output")
    p.add_argument("--output_file", type=str, default=None, help="Path to save terminal ANSI output")
    p.add_argument(
        "--no_boundaries", action="store_true",
        help="Hide token boundary markers in terminal output",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.text is None and args.input_file is None:
        print("Provide --text or --input_file", file=sys.stderr)
        sys.exit(1)

    text = args.text
    if args.input_file:
        text = Path(args.input_file).read_text("utf-8")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    spans = tokenize_to_spans(text, tokenizer)
    colours = get_colours(spans, args.colour_mode, tokenizer.vocab_size)

    # Terminal output
    ansi = render_terminal(spans, colours, show_boundaries=not args.no_boundaries)
    print(ansi)

    if args.output_file:
        Path(args.output_file).write_text(ansi, "utf-8", errors="replace")
        print(f"Saved terminal output -> {args.output_file}")

    # HTML output
    if args.html:
        html_content = render_html(text, spans, colours)
        Path(args.html).write_text(html_content, "utf-8")
        print(f"Saved interactive HTML -> {args.html}")


if __name__ == "__main__":
    main()
