#!/usr/bin/env python3
"""Create an interactive PyVis network from first-association probability YAML.

Nodes are tokens. Directed edges represent next-token associations from each
start-token row. Edge thickness is proportional to association strength.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _build_edges(start_tokens: Sequence[int], probs: np.ndarray, top_k: int, min_strength: float) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    vocab_size = probs.shape[1]
    k = max(1, min(top_k, vocab_size))
    for row_idx, source in enumerate(start_tokens):
        row = probs[row_idx]
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        for tgt in top_idx:
            strength = float(row[tgt])
            if strength >= min_strength:
                edges.append((int(source), int(tgt), strength))
    return edges


def _inject_controls(
    html_text: str,
    start_tokens: Sequence[int],
    radios: Sequence[Tuple[int, str]],
    initial_mode: str,
) -> str:
    selector_html: List[str] = []
    for token_id, display in radios:
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
    <strong>Manual node selection (checkbox):</strong><br/>
    <button id="add-node-btn" type="button">Add checked nodes</button>
    <button id="remove-node-btn" type="button">Remove checked nodes</button>
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

  function startMode() {{
    const el = document.querySelector('input[name="start_mode"]:checked');
    return el ? el.value : 'none';
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

  function updateVisibility() {{
    const shown = computeShownNodes();

    nodes.forEach((n) => {{
      nodes.update({{ id: n.id, hidden: !shown.has(n.id) }});
    }});

    edges.forEach((e) => {{
      const visible = shown.has(e.from) && shown.has(e.to);
      edges.update({{ id: e.id, hidden: !visible }});
    }});

    refreshShownList(shown);
  }}

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

    start_tokens = [int(x) for x in data["start_tokens"]]
    probs = np.asarray(data["probabilities"], dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"Expected probabilities shape [num_start_tokens, vocab_size], got {probs.shape}")
    if probs.shape[0] != len(start_tokens):
        raise ValueError("start_tokens length does not match probabilities rows")

    vocab_size = int(probs.shape[1])
    vocab_labels = _to_vocab_label_map(data, vocab_size)
    edges = _build_edges(start_tokens, probs, args.top_k, args.min_strength)

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
        )

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(args.output_html))

    html_text = args.output_html.read_text(encoding="utf-8")

    if args.node_selector_scope == "start":
        radio_ids = sorted(set(start_tokens))[: args.node_selector_limit]
    else:
        radio_ids = sorted(all_nodes)[: args.node_selector_limit]

    radios = [(tid, _label(tid, vocab_labels)) for tid in radio_ids]
    html_text = _inject_controls(html_text, start_tokens, radios, args.initial_start_tokens)
    args.output_html.write_text(html_text, encoding="utf-8")

    print(f"Wrote interactive graph to {args.output_html}")


if __name__ == "__main__":
    main()
