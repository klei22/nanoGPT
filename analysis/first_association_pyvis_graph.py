#!/usr/bin/env python3
"""Create an interactive PyVis network from first-association probability YAML.

Nodes are tokens. Directed edges represent next-token associations from each
start-token row. Edge thickness is proportional to association strength.

Compute-reduction features:
- static (no physics) graph rendering
- optional start-token subsampling
- optional global edge pruning by percentile / max-edges
- optional torch CUDA top-k extraction backend
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from pyvis.network import Network


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive PyVis graph from first-association probabilities")
    p.add_argument("--probs_yaml", required=True, type=Path, help="Path to probs_<label>.yaml")
    p.add_argument("--output_html", required=True, type=Path, help="Output interactive HTML graph")
    p.add_argument("--top_k", type=int, default=20, help="Top-k outgoing edges per start token")
    p.add_argument("--min_strength", type=float, default=0.0, help="Minimum probability strength to keep an edge")
    p.add_argument(
        "--backend",
        choices=["auto", "numpy", "torch_cuda"],
        default="auto",
        help="Top-k extraction backend. 'torch_cuda' uses GPU if torch+CUDA are available.",
    )
    p.add_argument("--max_start_tokens", type=int, default=None, help="Optional cap on number of start-token rows processed")
    p.add_argument("--start_token_stride", type=int, default=1, help="Subsample start-token rows by stride before edge extraction")
    p.add_argument("--max_edges", type=int, default=12000, help="Cap final number of edges (largest strengths kept)")
    p.add_argument("--edge_percentile_keep", type=float, default=None, help="Keep only edges >= this global strength percentile (0-100)")
    p.add_argument(
        "--initial_start_tokens",
        choices=["none", "all"],
        default="none",
        help="Initial inclusion of start-token nodes in the visible graph",
    )
    p.add_argument(
        "--node_selector_scope",
        choices=["start", "all"],
        default="start",
        help="Which tokens appear in manual node checkbox list",
    )
    p.add_argument("--node_selector_limit", type=int, default=256, help="Max number of checkbox entries rendered")
    p.add_argument("--height", type=str, default="900px")
    p.add_argument("--width", type=str, default="100%")
    return p.parse_args()


def _load_prob_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {path}")
    if "start_tokens" not in data or "probabilities" not in data:
        raise ValueError(f"{path} must contain 'start_tokens' and 'probabilities'")
    return data


def _to_vocab_label_map(data: Dict[str, Any], vocab_size: int) -> Dict[int, str]:
    raw = data.get("vocab_labels", {})
    out: Dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[int(k)] = str(v)
    elif isinstance(raw, list):
        for i, v in enumerate(raw[:vocab_size]):
            out[i] = str(v)
    return out


def _label(token_id: int, vocab: Dict[int, str]) -> str:
    txt = vocab.get(int(token_id), str(int(token_id)))
    esc = str(txt).replace("\n", "\\n").replace("\t", "\\t")
    return f"{int(token_id)}:{esc}"


def _subsample_start_rows(
    start_tokens: Sequence[int],
    probs: np.ndarray,
    stride: int,
    max_start_tokens: Optional[int],
) -> Tuple[List[int], np.ndarray]:
    if stride < 1:
        raise ValueError("--start_token_stride must be >= 1")
    idx = np.arange(len(start_tokens))[::stride]
    if max_start_tokens is not None and max_start_tokens > 0:
        idx = idx[:max_start_tokens]
    st = [int(start_tokens[i]) for i in idx.tolist()]
    return st, probs[idx, :]


def _build_edges_numpy(start_tokens: Sequence[int], probs: np.ndarray, top_k: int, min_strength: float) -> List[Tuple[int, int, float]]:
    vocab_size = probs.shape[1]
    k = max(1, min(top_k, vocab_size))

    top_idx = np.argpartition(probs, -k, axis=1)[:, -k:]
    top_vals = np.take_along_axis(probs, top_idx, axis=1)
    order = np.argsort(-top_vals, axis=1)
    sorted_idx = np.take_along_axis(top_idx, order, axis=1)
    sorted_vals = np.take_along_axis(top_vals, order, axis=1)

    edges: List[Tuple[int, int, float]] = []
    for row_i, src in enumerate(start_tokens):
        for col_i, tgt in enumerate(sorted_idx[row_i]):
            strength = float(sorted_vals[row_i, col_i])
            if strength >= min_strength:
                edges.append((int(src), int(tgt), strength))
    return edges


def _build_edges_torch_cuda(start_tokens: Sequence[int], probs: np.ndarray, top_k: int, min_strength: float) -> List[Tuple[int, int, float]]:
    try:
        import torch
    except Exception:
        print("torch not available; falling back to numpy backend")
        return _build_edges_numpy(start_tokens, probs, top_k, min_strength)

    if not torch.cuda.is_available():
        print("CUDA not available; falling back to numpy backend")
        return _build_edges_numpy(start_tokens, probs, top_k, min_strength)

    t = torch.as_tensor(probs, dtype=torch.float32, device="cuda")
    k = max(1, min(top_k, t.size(1)))
    vals, idx = torch.topk(t, k=k, dim=1, largest=True, sorted=True)
    vals_np = vals.cpu().numpy()
    idx_np = idx.cpu().numpy()

    edges: List[Tuple[int, int, float]] = []
    for row_i, src in enumerate(start_tokens):
        for col_i, tgt in enumerate(idx_np[row_i]):
            strength = float(vals_np[row_i, col_i])
            if strength >= min_strength:
                edges.append((int(src), int(tgt), strength))
    return edges


def _build_edges(
    start_tokens: Sequence[int],
    probs: np.ndarray,
    top_k: int,
    min_strength: float,
    backend: str,
) -> List[Tuple[int, int, float]]:
    selected = backend
    if backend == "auto":
        selected = "torch_cuda"
    if selected == "torch_cuda":
        return _build_edges_torch_cuda(start_tokens, probs, top_k, min_strength)
    return _build_edges_numpy(start_tokens, probs, top_k, min_strength)


def _prune_edges(
    edges: List[Tuple[int, int, float]],
    edge_percentile_keep: Optional[float],
    max_edges: Optional[int],
) -> List[Tuple[int, int, float]]:
    if not edges:
        return edges

    pruned = edges
    if edge_percentile_keep is not None:
        if not (0.0 <= edge_percentile_keep <= 100.0):
            raise ValueError("--edge_percentile_keep must be within [0, 100]")
        strengths = np.array([s for _, _, s in pruned], dtype=np.float64)
        threshold = float(np.percentile(strengths, edge_percentile_keep))
        pruned = [e for e in pruned if e[2] >= threshold]

    if max_edges is not None and max_edges > 0 and len(pruned) > max_edges:
        pruned = sorted(pruned, key=lambda e: e[2], reverse=True)[:max_edges]

    return pruned


def _inject_controls(
    html_text: str,
    start_tokens: Sequence[int],
    selectors: Sequence[Tuple[int, str]],
    initial_mode: str,
) -> str:
    selector_html: List[str] = []
    for token_id, display in selectors:
        selector_html.append(
            f'<label><input type="checkbox" name="node_select" value="{token_id}"> {display}</label><br/>'
        )

    controls = f"""
<div id="fa-controls" style="position:fixed; right:10px; top:10px; width:340px; max-height:92vh; overflow:auto; background:#ffffffee; padding:10px; border:1px solid #ccc; z-index:9999; font-family:Arial,sans-serif; font-size:13px;">
  <h3 style="margin:0 0 8px 0;">First-Association Graph Controls</h3>
  <div style="margin-bottom:8px;">
    <strong>Start-token nodes:</strong><br/>
    <label><input type="radio" name="start_mode" value="none" {'checked' if initial_mode=='none' else ''}> none</label>
    <label><input type="radio" name="start_mode" value="all" {'checked' if initial_mode=='all' else ''}> all</label>
  </div>
  <div style="margin-bottom:8px;">
    <strong>Edge strength threshold:</strong><br/>
    <input id="edge-threshold-slider" type="range" min="0" max="1" step="0.001" value="0.0" style="width:180px; vertical-align:middle;" />
    <input id="edge-threshold-input" type="number" min="0" max="1" step="0.001" value="0.0" style="width:72px;" />
  </div>
  <div style="margin-bottom:8px;">
    <strong>Manual node selection (checkbox):</strong><br/>
    <div style="margin-bottom:6px;">
      <button id="select-all-btn" type="button">Select all</button>
      <button id="clear-all-btn" type="button">Clear all</button>
    </div>
    <div style="margin-bottom:6px;">
      <input id="regex-input" type="text" placeholder="regex for checkbox labels (e.g. ^12:|king)" style="width:220px;" />
      <button id="regex-select-btn" type="button">Select regex matches</button>
    </div>
    <button id="add-node-btn" type="button">Add checked nodes</button>
    <button id="remove-node-btn" type="button">Remove checked nodes</button>
    <button id="straighten-edges-btn" type="button">Straighten shown edges</button>
    <div style="margin-top:6px;">
      <select id="organize-layout-select">
        <option value="circle">Organize: circle</option>
        <option value="grid">Organize: grid</option>
        <option value="line">Organize: line</option>
        <option value="radial">Organize: radial by degree</option>
      </select>
      <button id="organize-layout-btn" type="button">Apply organize</button>
    </div>
    <div id="node-selector" style="max-height:320px; overflow:auto; border:1px solid #ddd; padding:6px; margin-top:6px;">{''.join(selector_html)}</div>
  </div>
  <div>
    <strong>Shown nodes</strong>
    <ul id="shown-nodes" style="max-height:220px; overflow:auto; border:1px solid #ddd; padding:6px; margin-top:6px;"></ul>
  </div>
</div>
<script>
(function() {{
  const startTokens = new Set({json.dumps([int(x) for x in start_tokens])});
  const manualNodes = new Set();

  function selectedCheckboxTokens() {{
    const selected = [];
    document.querySelectorAll('input[name="node_select"]:checked').forEach((el) => {{
      selected.push(parseInt(el.value, 10));
    }});
    return selected;
  }}

  function allNodeCheckboxes() {{
    return Array.from(document.querySelectorAll('input[name="node_select"]'));
  }}

  function checkboxLabelText(cb) {{
    const parent = cb.parentElement;
    if (!parent) return cb.value;
    return (parent.textContent || '').trim();
  }}

  function startMode() {{
    const el = document.querySelector('input[name="start_mode"]:checked');
    return el ? el.value : 'none';
  }}

  function edgeThreshold() {{
    const input = document.getElementById('edge-threshold-input');
    const parsed = parseFloat(input.value);
    if (Number.isNaN(parsed)) return 0.0;
    return Math.min(1.0, Math.max(0.0, parsed));
  }}

  function computeShownNodes() {{
    const shown = new Set(manualNodes);
    if (startMode() === 'all') {{
      startTokens.forEach((x) => shown.add(x));
    }}
    return shown;
  }}

  function refreshShownList(shown) {{
    const ul = document.getElementById('shown-nodes');
    ul.innerHTML = '';
    Array.from(shown).sort((a,b) => a-b).forEach((id) => {{
      const li = document.createElement('li');
      const n = nodes.get(id);
      li.textContent = n ? `${{id}}: ${{n.label}}` : String(id);
      ul.appendChild(li);
    }});
  }}

  function organizeShownNodes(layoutMode) {{
    const shownNodes = [];
    nodes.forEach((n) => {{
      if (!n.hidden) shownNodes.push(n.id);
    }});
    if (!shownNodes.length) return;

    if (layoutMode === 'circle' || layoutMode === 'radial') {{
      let ordered = shownNodes.slice();
      if (layoutMode === 'radial') {{
        const degree = new Map();
        shownNodes.forEach((id) => degree.set(id, 0));
        edges.forEach((e) => {{
          if (e.hidden) return;
          if (degree.has(e.from)) degree.set(e.from, degree.get(e.from) + 1);
          if (degree.has(e.to)) degree.set(e.to, degree.get(e.to) + 1);
        }});
        ordered.sort((a, b) => (degree.get(b) - degree.get(a)) || (a - b));
      }} else {{
        ordered.sort((a, b) => a - b);
      }}
      const radius = Math.max(160, 28 * ordered.length / (2 * Math.PI));
      ordered.forEach((id, idx) => {{
        const theta = (2 * Math.PI * idx) / ordered.length;
        nodes.update({{ id, x: radius * Math.cos(theta), y: radius * Math.sin(theta), fixed: {{x: true, y: true}} }});
      }});
    }} else if (layoutMode === 'grid') {{
      const ordered = shownNodes.slice().sort((a, b) => a - b);
      const cols = Math.ceil(Math.sqrt(ordered.length));
      const step = 90;
      ordered.forEach((id, idx) => {{
        const r = Math.floor(idx / cols);
        const c = idx % cols;
        nodes.update({{ id, x: c * step, y: r * step, fixed: {{x: true, y: true}} }});
      }});
    }} else if (layoutMode === 'line') {{
      const ordered = shownNodes.slice().sort((a, b) => a - b);
      const step = 70;
      ordered.forEach((id, idx) => {{
        nodes.update({{ id, x: idx * step, y: 0, fixed: {{x: true, y: true}} }});
      }});
    }}

    if (typeof network !== 'undefined' && network.fit) {{
      network.fit();
    }}
  }}

  function updateVisibility() {{
    const shown = computeShownNodes();
    const threshold = edgeThreshold();
    const activeNodes = new Set();

    edges.forEach((e) => {{
      const strength = typeof e.strength === 'number' ? e.strength : 0.0;
      const visible = shown.has(e.from) && shown.has(e.to) && strength >= threshold;
      if (visible) {{
        activeNodes.add(e.from);
        activeNodes.add(e.to);
      }}
      edges.update({{ id: e.id, hidden: !visible }});
    }});

    nodes.forEach((n) => {{
      const visible = shown.has(n.id) && activeNodes.has(n.id);
      nodes.update({{ id: n.id, hidden: !visible }});
    }});

    refreshShownList(activeNodes);
  }}

  document.getElementById('select-all-btn').addEventListener('click', () => {{
    allNodeCheckboxes().forEach((cb) => {{
      cb.checked = true;
    }});
  }});

  document.getElementById('clear-all-btn').addEventListener('click', () => {{
    allNodeCheckboxes().forEach((cb) => {{
      cb.checked = false;
    }});
  }});

  document.getElementById('regex-select-btn').addEventListener('click', () => {{
    const pattern = document.getElementById('regex-input').value;
    if (!pattern) return;
    let rx;
    try {{
      rx = new RegExp(pattern);
    }} catch (err) {{
      alert(`Invalid regex: ${{err}}`);
      return;
    }}
    allNodeCheckboxes().forEach((cb) => {{
      cb.checked = rx.test(checkboxLabelText(cb));
    }});
  }});

  document.getElementById('add-node-btn').addEventListener('click', () => {{
    const toks = selectedCheckboxTokens();
    if (!toks.length) return;
    toks.forEach((tok) => manualNodes.add(tok));
    updateVisibility();
  }});

  document.getElementById('remove-node-btn').addEventListener('click', () => {{
    const toks = selectedCheckboxTokens();
    if (!toks.length) return;
    toks.forEach((tok) => manualNodes.delete(tok));
    updateVisibility();
  }});

  document.getElementById('straighten-edges-btn').addEventListener('click', () => {{
    edges.forEach((e) => {{
      if (e.hidden) return;
      edges.update({{ id: e.id, smooth: false }});
    }});
  }});

  document.getElementById('organize-layout-btn').addEventListener('click', () => {{
    const mode = document.getElementById('organize-layout-select').value;
    organizeShownNodes(mode);
  }});

  const thresholdSlider = document.getElementById('edge-threshold-slider');
  const thresholdInput = document.getElementById('edge-threshold-input');

  thresholdSlider.addEventListener('input', () => {{
    thresholdInput.value = thresholdSlider.value;
    updateVisibility();
  }});

  thresholdInput.addEventListener('input', () => {{
    let v = parseFloat(thresholdInput.value);
    if (Number.isNaN(v)) v = 0.0;
    v = Math.min(1.0, Math.max(0.0, v));
    thresholdInput.value = v.toFixed(3);
    thresholdSlider.value = thresholdInput.value;
    updateVisibility();
  }});

  document.querySelectorAll('input[name="start_mode"]').forEach((el) => {{
    el.addEventListener('change', updateVisibility);
  }});

  updateVisibility();
}})();
</script>
"""

    return html_text.replace("</body>", controls + "\n</body>")


def main() -> None:
    args = parse_args()
    data = _load_prob_yaml(args.probs_yaml)

    start_tokens_all = [int(x) for x in data["start_tokens"]]
    probs_all = np.asarray(data["probabilities"], dtype=np.float64)
    if probs_all.ndim != 2:
        raise ValueError(f"Expected probabilities shape [num_start_tokens, vocab_size], got {probs_all.shape}")
    if probs_all.shape[0] != len(start_tokens_all):
        raise ValueError("start_tokens length does not match probabilities rows")

    start_tokens, probs = _subsample_start_rows(
        start_tokens_all,
        probs_all,
        stride=args.start_token_stride,
        max_start_tokens=args.max_start_tokens,
    )

    vocab_size = int(probs.shape[1])
    vocab_labels = _to_vocab_label_map(data, vocab_size)

    edges = _build_edges(start_tokens, probs, args.top_k, args.min_strength, args.backend)
    edges = _prune_edges(edges, args.edge_percentile_keep, args.max_edges)

    print(
        f"Graph reduction summary: start_tokens={len(start_tokens)}/{len(start_tokens_all)}, "
        f"edges={len(edges)}"
    )

    net = Network(height=args.height, width=args.width, directed=True, bgcolor="#ffffff", font_color="#222222")
    net.toggle_physics(False)

    label_name = str(data.get("label", "model"))
    all_nodes = set(start_tokens)
    for src, tgt, _ in edges:
        all_nodes.add(src)
        all_nodes.add(tgt)

    for node_id in sorted(all_nodes):
        net.add_node(
            node_id,
            label=_label(node_id, vocab_labels),
            title=f"token {node_id}",
            shape="dot",
            size=12,
        )

    max_strength = max((s for _, _, s in edges), default=1.0)
    for edge_idx, (src, tgt, strength) in enumerate(edges):
        width = 1.0 + 8.0 * (strength / max_strength if max_strength > 0 else 0.0)
        net.add_edge(
            src,
            tgt,
            id=f"e{edge_idx}",
            value=width,
            width=width,
            title=f"{label_name}: {strength:.6f}",
            color={"opacity": 0.7},
            strength=float(strength),
        )

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(args.output_html))

    html_text = args.output_html.read_text(encoding="utf-8")

    if args.node_selector_scope == "start":
        selector_ids = sorted(set(start_tokens))[: args.node_selector_limit]
    else:
        selector_ids = sorted(all_nodes)[: args.node_selector_limit]

    selectors = [(tid, _label(tid, vocab_labels)) for tid in selector_ids]
    html_text = _inject_controls(html_text, start_tokens, selectors, args.initial_start_tokens)
    args.output_html.write_text(html_text, encoding="utf-8")

    print(f"Wrote interactive graph to {args.output_html}")


if __name__ == "__main__":
    main()
