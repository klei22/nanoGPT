#!/usr/bin/env python3
"""
spm_vocab_freq_dashboard.py

Build a single self-contained HTML dashboard (Plotly + vanilla JS) that shows:

LEFT:
  - SentencePiece vocab tokens + total frequency across *all* .txt files in a directory
  - searchable dropdown to pick a token (and optional click-to-select from table)

RIGHT:
  - per-file counts for the currently-selected token (bar chart)
  - updates live in the same HTML (no server)

Why we require a .model:
  - SentencePiece tokenization is not plain substring matching; to get true token frequencies,
    we MUST encode text using the SentencePiece model.

Defaults:
  --vocab trained_spm_model.vocab
  --model inferred by replacing ".vocab" -> ".model" if not provided
  --dir   required

Output:
  vocab_freq_dashboard.html (or --out)

Example:
  python3 spm_vocab_freq_dashboard.py --dir ./text --vocab trained_spm_model.vocab
  python3 spm_vocab_freq_dashboard.py --dir ./text --model trained_spm_model.model --top-k 2000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm


def infer_model_path_from_vocab(vocab_path: Path) -> Path:
    # trained_spm_model.vocab -> trained_spm_model.model
    if vocab_path.suffix.lower() == ".vocab":
        return vocab_path.with_suffix(".model")
    # fallback: append .model
    return Path(str(vocab_path) + ".model")


def iter_text_files(root: Path, recursive: bool, suffixes: Tuple[str, ...]) -> List[Path]:
    if recursive:
        it = root.rglob("*")
    else:
        it = root.glob("*")
    files = []
    for p in it:
        if p.is_file() and p.suffix.lower() in suffixes:
            files.append(p)
    files.sort()
    return files


def count_tokens_in_file(sp: spm.SentencePieceProcessor, path: Path) -> Counter:
    """
    Streaming-ish: encode line by line to avoid loading huge files into memory.
    """
    c = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ids = sp.encode(line, out_type=int)
            c.update(ids)
    return c


def human_token(tok: str) -> str:
    # Make SentencePiece boundary visible and avoid crazy HTML rendering.
    # Keep it readable: ▁ (U+2581) is the "word boundary" marker in SPM.
    return tok.replace("\t", " ").replace("\n", "\\n")


def build_html(
    title: str,
    token_rows: List[Dict],
    per_file_counts: Dict[str, Dict[str, int]],
    file_order: List[str],
    default_token_id: int,
    out_path: Path,
) -> None:
    """
    token_rows: list of dicts for top-k tokens: {id, token, count}
    per_file_counts: { token_id(str) -> { file_name -> count } } only for tokens we embed
    file_order: stable order of files for bar chart
    """
    payload = {
        "title": title,
        "tokens": token_rows,
        "per_file": per_file_counts,
        "files": file_order,
        "default_token_id": default_token_id,
    }

    # NOTE: This uses Plotly CDN for a "dynamic" interactive page without extra deps.
    # If you need fully offline HTML (no CDN), we can embed plotly.min.js, but the file is huge.
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Noto Sans", Arial, sans-serif;
      margin: 0; padding: 0;
    }}
    header {{
      padding: 12px 16px;
      border-bottom: 1px solid #ddd;
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    header h1 {{
      font-size: 16px;
      margin: 0;
      flex: 1;
    }}
    .container {{
      display: flex;
      height: calc(100vh - 58px);
      gap: 10px;
      padding: 10px;
      box-sizing: border-box;
    }}
    .panel {{
      flex: 1;
      min-width: 380px;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px;
      box-sizing: border-box;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .panel h2 {{
      font-size: 14px;
      margin: 0 0 8px 0;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .controls label {{
      font-size: 12px;
      color: #333;
    }}
    select, input {{
      font-size: 12px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid #ccc;
      outline: none;
    }}
    #tableDiv {{
      flex: 1;
      min-height: 200px;
    }}
    #barDiv {{
      flex: 1;
      min-height: 200px;
    }}
    .note {{
      font-size: 12px;
      color: #555;
      padding: 8px 0 0 0;
    }}
    .muted {{
      color: #777;
    }}
  </style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <span class="muted">Click a row (or use dropdown) to update per-file counts.</span>
</header>

<div class="container">
  <div class="panel">
    <h2>Vocab + total frequency (directory aggregate)</h2>
    <div class="controls">
      <label for="tokenSelect">Token:</label>
      <select id="tokenSelect"></select>
      <label for="searchBox">Search:</label>
      <input id="searchBox" type="text" placeholder="type to filter tokens..." />
    </div>
    <div id="tableDiv"></div>
    <div class="note">
      Showing <b>top {len(token_rows)}</b> tokens by frequency.
    </div>
  </div>

  <div class="panel">
    <h2 id="rightTitle">Per-file counts</h2>
    <div id="barDiv"></div>
    <div class="note">
      Bars show token count per file (same tokenization as training).
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};

function fmtTokenRow(t) {{
  // Make whitespace visible-ish in dropdown
  let s = t.token;
  // show the word-boundary marker as "▁" (already is), but keep readable
  if (s.length > 60) s = s.slice(0, 57) + "…";
  return `${{t.id}}: ${{s}} (${{t.count}})`;
}}

function buildSelectOptions(tokens) {{
  const sel = document.getElementById("tokenSelect");
  sel.innerHTML = "";
  for (const t of tokens) {{
    const opt = document.createElement("option");
    opt.value = String(t.id);
    opt.textContent = fmtTokenRow(t);
    sel.appendChild(opt);
  }}
}}

function filterTokens(tokens, q) {{
  if (!q) return tokens;
  q = q.toLowerCase();
  return tokens.filter(t => String(t.id).includes(q) || (t.token || "").toLowerCase().includes(q));
}}

function renderTable(tokens) {{
  // Plotly table
  const ids = tokens.map(t => t.id);
  const toks = tokens.map(t => t.token);
  const counts = tokens.map(t => t.count);

  const tableData = [{{
    type: "table",
    header: {{
      values: ["<b>ID</b>", "<b>Token</b>", "<b>Total Count</b>"],
      align: ["right", "left", "right"],
    }},
    cells: {{
      values: [ids, toks, counts],
      align: ["right", "left", "right"],
      height: 22
    }}
  }}];

  const layout = {{
    margin: {{l: 10, r: 10, t: 10, b: 10}},
  }};

  Plotly.newPlot("tableDiv", tableData, layout, {{displayModeBar: false}});

  // Click-to-select token: for tables, plotly_click gives pointNumber (row index)
  const tableDiv = document.getElementById("tableDiv");
  tableDiv.on("plotly_click", (ev) => {{
    try {{
      const row = ev.points[0].pointNumber;
      const tok = tokens[row];
      if (tok) {{
        selectToken(String(tok.id), true);
      }}
    }} catch (e) {{}}
  }});
}}

function renderBar(tokenId) {{
  const tok = DATA.tokens.find(t => String(t.id) === String(tokenId));
  const name = tok ? tok.token : `(id=${{tokenId}})`;

  const per = DATA.per_file[String(tokenId)] || {{}};
  const xs = DATA.files.slice();
  const ys = xs.map(fn => (per[fn] || 0));

  const trace = {{
    type: "bar",
    x: xs,
    y: ys
  }};

  const layout = {{
    margin: {{l: 50, r: 10, t: 30, b: 120}},
    xaxis: {{
      tickangle: 35,
      automargin: true
    }},
    yaxis: {{
      title: "Count"
    }},
    title: `Token: ${{name}} (id=${{tokenId}})`
  }};

  Plotly.newPlot("barDiv", [trace], layout, {{displayModeBar: true}});
  document.getElementById("rightTitle").textContent = "Per-file counts";
}}

function selectToken(tokenId, updateSelect) {{
  if (updateSelect) {{
    const sel = document.getElementById("tokenSelect");
    sel.value = String(tokenId);
  }}
  renderBar(tokenId);
}}

function init() {{
  buildSelectOptions(DATA.tokens);

  // Default selection
  const sel = document.getElementById("tokenSelect");
  sel.value = String(DATA.default_token_id);
  renderTable(DATA.tokens);
  renderBar(DATA.default_token_id);

  sel.addEventListener("change", (e) => {{
    selectToken(e.target.value, false);
  }});

  const search = document.getElementById("searchBox");
  search.addEventListener("input", (e) => {{
    const q = e.target.value || "";
    const filtered = filterTokens(DATA.tokens, q);

    // update dropdown to filtered list, but keep current selection if still present
    const cur = document.getElementById("tokenSelect").value;
    buildSelectOptions(filtered);

    // if current selection is in filtered list, keep it; else select first
    const hasCur = filtered.some(t => String(t.id) === String(cur));
    const newId = hasCur ? cur : (filtered.length ? String(filtered[0].id) : String(DATA.default_token_id));
    document.getElementById("tokenSelect").value = newId;

    // rerender table with filtered tokens
    renderTable(filtered);

    // update bar based on dropdown selection
    selectToken(newId, false);
  }});
}}

init();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="trained_spm_model.vocab",
                    help="SentencePiece vocab file (default: trained_spm_model.vocab). Used to infer .model if --model not given.")
    ap.add_argument("--model", default=None,
                    help="SentencePiece model file (.model). If omitted, inferred from --vocab by replacing .vocab -> .model.")
    ap.add_argument("--dir", required=True,
                    help="Directory of text files to scan.")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories (default: false).")
    ap.add_argument("--suffixes", default=".txt",
                    help="Comma-separated suffixes to include (default: .txt). Example: .txt,.md")
    ap.add_argument("--top-k", type=int, default=1500,
                    help="Embed only top-K tokens by total frequency into the HTML for interactivity (default: 1500).")
    ap.add_argument("--out", default="vocab_freq_dashboard.html",
                    help="Output HTML path (default: vocab_freq_dashboard.html)")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Only consider tokens with total count >= this (default: 1).")
    args = ap.parse_args()

    vocab_path = Path(args.vocab)
    model_path = Path(args.model) if args.model else infer_model_path_from_vocab(vocab_path)
    root = Path(args.dir)
    out = Path(args.out)

    if not model_path.exists():
        raise SystemExit(f"SentencePiece model not found: {model_path} (pass --model or ensure it matches --vocab)")
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    suffixes = tuple(s.strip().lower() for s in args.suffixes.split(",") if s.strip())
    files = iter_text_files(root, recursive=args.recursive, suffixes=suffixes)
    if not files:
        raise SystemExit(f"No files found in {root} with suffixes={suffixes} (try --recursive or --suffixes)")

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()

    print(f"[info] model: {model_path}")
    print(f"[info] vocab size: {vocab_size}")
    print(f"[info] scanning {len(files)} files under: {root}")

    total = Counter()
    per_file: Dict[str, Counter] = {}
    file_names: List[str] = []

    for p in files:
        rel = str(p.relative_to(root))
        file_names.append(rel)
        c = count_tokens_in_file(sp, p)
        per_file[rel] = c
        total.update(c)

    # Build top tokens (by total frequency)
    items = [(tid, cnt) for tid, cnt in total.items() if cnt >= args.min_count]
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        raise SystemExit(f"No tokens met min-count={args.min_count} (unexpected).")

    top_items = items[: max(1, args.top_k)]

    token_rows: List[Dict] = []
    # per_file_counts: token_id(str) -> file -> count  (only for embedded tokens)
    per_file_counts: Dict[str, Dict[str, int]] = {}

    for tid, cnt in top_items:
        tok = human_token(sp.id_to_piece(int(tid)))
        token_rows.append({"id": int(tid), "token": tok, "count": int(cnt)})

    # Build per-file map for embedded tokens
    for tid, _ in top_items:
        tid = int(tid)
        k = str(tid)
        per_file_counts[k] = {}
        for fn in file_names:
            v = per_file[fn].get(tid, 0)
            if v:
                per_file_counts[k][fn] = int(v)

    default_token_id = int(top_items[0][0])

    title = f"SentencePiece token frequency dashboard ({root.name})"
    build_html(
        title=title,
        token_rows=token_rows,
        per_file_counts=per_file_counts,
        file_order=file_names,
        default_token_id=default_token_id,
        out_path=out,
    )

    print(f"[done] wrote: {out}")
    print(f"[note] Uses Plotly CDN for interactivity; open the HTML in your browser.")


if __name__ == "__main__":
    main()

