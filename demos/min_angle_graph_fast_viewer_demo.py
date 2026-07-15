#!/usr/bin/env python3
"""Precompile exported min-angle CSVs into a fast HTML graph viewer.

The recursive angle webapp can already export adjacency/token-list CSV files.
This tool converts those existing CSV artifacts into a self-contained HTML page
that keeps the same dark rectangular node/angle-labelled edge presentation, but
renders the graph through a single Canvas 2D surface instead of maintaining a
large SVG/Plotly object tree. It can still generate synthetic data for demos and
can optionally pre-render frames in parallel for video assembly.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Fast Recursive Angle Graph</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
  body { margin: 0; background: #020617; color: #e2e8f0; }
  header { padding: 12px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; background: #0f172a; border-bottom: 1px solid #1e293b; }
  button, input { accent-color: #38bdf8; }
  button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 7px 10px; cursor: pointer; }
  button:hover { background: #334155; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; min-height: calc(100vh - 58px); }
  #stage { position: relative; overflow: hidden; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  aside { border-left: 1px solid #1e293b; padding: 14px; background: #0b1120; overflow: auto; }
  .stat { display: grid; grid-template-columns: 1fr auto; gap: 8px; padding: 5px 0; border-bottom: 1px solid #172033; }
  .muted, .selection { color: #94a3b8; }
  #tip { position: absolute; pointer-events: none; display: none; max-width: 380px; padding: 8px 10px; border: 1px solid #334155; border-radius: 8px; background: rgba(15,23,42,.94); box-shadow: 0 8px 30px rgba(0,0,0,.4); font-size: 12px; white-space: pre-wrap; }
  .pill { border: 1px solid #334155; border-radius: 999px; padding: 4px 8px; color: #cbd5e1; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border-bottom: 1px solid #172033; padding: 4px 6px; text-align: left; }
  code { color: #cbd5e1; }
</style>
</head>
<body>
<header>
  <strong>Fast Recursive Angle Graph</strong>
  <button id="play">Pause</button>
  <label>Frame <input id="frame" type="range" min="0" max="0" value="0" /></label>
  <label><input id="edgeLabels" type="checkbox" checked /> angle labels</label>
  <label><input id="minOnly" type="checkbox" /> lowest-angle edges only</label>
  <span id="readout" class="pill"></span>
</header>
<main>
  <section id="stage"><canvas id="canvas"></canvas><div id="tip"></div></section>
  <aside>
    <h2>Precompiled CSV viewer</h2>
    <p class="selection">Nodes show token ID and token string. Drag nodes to reposition them; drag empty space to pan; mouse-wheel zooms; double-click resets the view. Edges are labelled with the angle between connected tokens.</p>
    <p class="muted">The source CSVs are precompiled into compact arrays once; interaction updates one canvas instead of thousands of SVG/Plotly elements.</p>
    <div id="stats"></div>
    __VIDEO_BLOCK__
    <h3>Hover detail</h3>
    <pre id="detail" class="muted">Move the mouse over a node.</pre>
    <h3>Token list</h3>
    <div style="max-height: 320px; overflow: auto"><table id="nodeTable"></table></div>
  </aside>
</main>
<script id="graph-data" type="application/json">__GRAPH_JSON__</script>
<script>
const graph = JSON.parse(document.getElementById('graph-data').textContent);
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false });
const stage = document.getElementById('stage');
const tip = document.getElementById('tip');
const frameSlider = document.getElementById('frame');
const readout = document.getElementById('readout');
const detail = document.getElementById('detail');
frameSlider.max = String(Math.max(0, graph.frames.length - 1));
const NODE_W = 150, NODE_H = 54;
let playing = graph.frames.length > 1, frame = 0, zoom = 1, panX = 0, panY = 0, drag = null, hover = -1, draggingNode = -1, nodeOffset = [0, 0];
document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
const minByNode = new Map();
for (const e of graph.edges) {
  for (const id of [e.source, e.target]) {
    const old = minByNode.get(id);
    if (old === undefined || e.angle_deg < old) minByNode.set(id, e.angle_deg);
  }
}
function isMinEdge(e) { return e.angle_deg <= (minByNode.get(e.source) ?? Infinity) + 1e-9 || e.angle_deg <= (minByNode.get(e.target) ?? Infinity) + 1e-9; }
function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1); const r = stage.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(r.width * dpr)); canvas.height = Math.max(1, Math.floor(r.height * dpr));
  canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px'; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
}
function coords() { return graph.frames[frame].xy; }
function project(x, y) { return [x * zoom + panX, y * zoom + panY]; }
function unproject(px, py) { return [(px - panX) / zoom, (py - panY) / zoom]; }
function drawRoundedRect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function draw() {
  const r = stage.getBoundingClientRect(); ctx.fillStyle = '#020617'; ctx.fillRect(0, 0, r.width, r.height);
  const c = coords(); const minOnly = document.getElementById('minOnly').checked; const edgeLabels = document.getElementById('edgeLabels').checked;
  ctx.lineCap = 'round'; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  for (const e of graph.edges) {
    const minEdge = isMinEdge(e); if (minOnly && !minEdge) continue;
    const [x1,y1] = project(c[e.source][0], c[e.source][1]); const [x2,y2] = project(c[e.target][0], c[e.target][1]);
    ctx.strokeStyle = minEdge ? '#fbbf24' : 'rgba(148, 163, 184, 0.52)'; ctx.globalAlpha = minEdge ? .98 : .72; ctx.lineWidth = minEdge ? 3.2 : 1.25;
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    if (edgeLabels && zoom > .35) {
      ctx.globalAlpha = 1; ctx.fillStyle = minEdge ? '#fde68a' : '#94a3b8'; ctx.strokeStyle = '#020617'; ctx.lineWidth = 3;
      const label = `${Number(e.angle_deg).toFixed(2)}°`; const lx = (x1 + x2) / 2, ly = (y1 + y2) / 2;
      ctx.strokeText(label, lx, ly); ctx.fillText(label, lx, ly);
    }
  }
  ctx.globalAlpha = 1;
  for (let i = 0; i < graph.nodes.length; i++) {
    const n = graph.nodes[i]; const [cx,cy] = project(c[i][0], c[i][1]); const x = cx - NODE_W/2, y = cy - NODE_H/2;
    drawRoundedRect(x, y, NODE_W, NODE_H, 12); ctx.fillStyle = '#111827'; ctx.fill(); ctx.strokeStyle = i === hover ? '#fbbf24' : '#38bdf8'; ctx.lineWidth = i === hover ? 2.3 : 1.1; ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillStyle = '#f8fafc'; ctx.font = '700 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'; ctx.fillText(String(n.token_id), cx, y + 20);
    ctx.fillStyle = '#cbd5e1'; ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'; ctx.fillText(String(n.token_raw ?? n.label ?? '').slice(0, 22), cx, y + 38);
  }
  readout.textContent = `frame ${frame + 1}/${graph.frames.length} · ${graph.nodes.length} nodes · ${graph.edges.length} edges`;
}
function nearestNode(ev) {
  const rect = canvas.getBoundingClientRect(); const [ux, uy] = unproject(ev.clientX - rect.left, ev.clientY - rect.top); const c = coords();
  for (let i = c.length - 1; i >= 0; i--) if (Math.abs(ux - c[i][0]) <= NODE_W/2 && Math.abs(uy - c[i][1]) <= NODE_H/2) return i;
  return -1;
}
canvas.addEventListener('mousemove', ev => {
  const rect = canvas.getBoundingClientRect();
  if (draggingNode >= 0) { const [ux, uy] = unproject(ev.clientX - rect.left, ev.clientY - rect.top); coords()[draggingNode] = [ux + nodeOffset[0], uy + nodeOffset[1]]; draw(); return; }
  if (drag) { panX += ev.clientX - drag.x; panY += ev.clientY - drag.y; drag = {x: ev.clientX, y: ev.clientY}; draw(); return; }
  hover = nearestNode(ev); draw();
  if (hover >= 0) { const n = graph.nodes[hover]; const text = JSON.stringify(n, null, 2); detail.textContent = text; tip.textContent = text; tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 14) + 'px'; }
  else { tip.style.display = 'none'; detail.textContent = 'Move the mouse over a node.'; }
});
canvas.addEventListener('mousedown', ev => { const rect = canvas.getBoundingClientRect(); const hit = nearestNode(ev); if (hit >= 0) { draggingNode = hit; const [ux, uy] = unproject(ev.clientX - rect.left, ev.clientY - rect.top); const p = coords()[hit]; nodeOffset = [p[0] - ux, p[1] - uy]; } else { drag = {x: ev.clientX, y: ev.clientY}; } });
window.addEventListener('mouseup', () => { drag = null; draggingNode = -1; });
canvas.addEventListener('wheel', ev => { ev.preventDefault(); const rect = canvas.getBoundingClientRect(); const mx = ev.clientX - rect.left, my = ev.clientY - rect.top; const [ux, uy] = unproject(mx, my); const factor = ev.deltaY < 0 ? 1.12 : .89; zoom = Math.max(.08, Math.min(8, zoom * factor)); const [px, py] = project(ux, uy); panX += mx - px; panY += my - py; draw(); }, {passive:false});
canvas.addEventListener('dblclick', () => { fitGraph(); draw(); });
frameSlider.addEventListener('input', () => { frame = Number(frameSlider.value); draw(); });
document.getElementById('edgeLabels').addEventListener('change', draw); document.getElementById('minOnly').addEventListener('change', draw);
document.getElementById('play').addEventListener('click', ev => { playing = !playing; ev.target.textContent = playing ? 'Pause' : 'Play'; });
function fitGraph() { const r = stage.getBoundingClientRect(); const c = coords(); const xs = c.map(p => p[0]), ys = c.map(p => p[1]); const minX = Math.min(...xs)-NODE_W, maxX = Math.max(...xs)+NODE_W, minY = Math.min(...ys)-NODE_H, maxY = Math.max(...ys)+NODE_H; zoom = Math.min(r.width / Math.max(1, maxX-minX), r.height / Math.max(1, maxY-minY)); zoom = Math.max(.08, Math.min(2, zoom)); panX = (r.width - (minX + maxX) * zoom) / 2; panY = (r.height - (minY + maxY) * zoom) / 2; }
function tick() { if (playing && graph.frames.length > 1) { frame = (frame + 1) % graph.frames.length; frameSlider.value = String(frame); draw(); } requestAnimationFrame(tick); }
document.getElementById('stats').innerHTML = Object.entries(graph.meta).map(([k,v]) => `<div class="stat"><span>${k}</span><strong>${v}</strong></div>`).join('');
document.getElementById('nodeTable').innerHTML = '<thead><tr><th>Token ID</th><th>Raw token</th><th>Connected</th></tr></thead><tbody>' + graph.nodes.slice(0, 500).map(n => `<tr><td>${n.token_id}</td><td><code>${String(n.token_raw ?? '').replace(/[&<>]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[s]))}</code></td><td>${n.connected_count ?? ''}</td></tr>`).join('') + '</tbody>';
window.addEventListener('resize', () => { resize(); fitGraph(); draw(); }); resize(); fitGraph(); draw(); tick();
</script>
</body>
</html>
"""

def _token_label(value: str) -> str:
    return value.replace("\\n", "\n")


def _node_id_column(row: dict[str, str]) -> str | None:
    for key in ("Token ID", "token_id", "id", "TokenID"):
        if key in row and row[key] != "":
            return row[key]
    return None


def read_token_list_csv(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    nodes: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_id = _node_id_column(row)
            if raw_id is None:
                continue
            token_id = int(float(raw_id))
            nodes[token_id] = {
                "token_id": token_id,
                "token_raw": _token_label(row.get("Raw token") or row.get("token_raw") or row.get("token") or str(token_id)),
                "token_display": row.get("Display") or row.get("token_display") or row.get("display") or "",
                "connected_count": int(float(row.get("Connected nodes") or row.get("connected_count") or row.get("degree") or 0)),
                "magnitude": float(row.get("Vector length") or row.get("magnitude") or 0.0),
            }
    return nodes


def read_dictionary_json(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    nodes: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        token_id = int(value.get("token_id", key))
        nodes[token_id] = {
            "token_id": token_id,
            "token_raw": value.get("token_raw") or value.get("raw") or value.get("token") or str(token_id),
            "token_display": value.get("token_display") or value.get("display") or "",
            "connected_count": int(value.get("connected_count", 0) or 0),
            "magnitude": float(value.get("magnitude", 0.0) or 0.0),
        }
    return nodes


def graph_from_adjacency_csv(
    adjacency_csv: Path,
    token_list_csv: Path | None = None,
    dictionary_json: Path | None = None,
) -> dict[str, Any]:
    """Compile exported recursive-group adjacency/token CSVs into viewer arrays."""
    nodes_by_id = read_dictionary_json(dictionary_json)
    nodes_by_id.update(read_token_list_csv(token_list_csv))
    with adjacency_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"{adjacency_csv} does not look like an adjacency matrix CSV")
    header_ids = [int(float(cell)) for cell in rows[0][1:] if cell != ""]
    ids = header_ids[:]
    edges_by_pair: dict[tuple[int, int], float] = {}
    degree: dict[int, int] = {token_id: 0 for token_id in ids}
    for row in rows[1:]:
        if not row or row[0] == "":
            continue
        source_id = int(float(row[0]))
        if source_id not in degree:
            ids.append(source_id)
            degree[source_id] = 0
        for target_id, cell in zip(header_ids, row[1:]):
            if source_id >= target_id or cell == "":
                continue
            try:
                angle = float(cell)
            except ValueError:
                continue
            pair = (source_id, target_id)
            edges_by_pair[pair] = angle
            degree[source_id] = degree.get(source_id, 0) + 1
            degree[target_id] = degree.get(target_id, 0) + 1
    ids = sorted(dict.fromkeys(ids))
    index = {token_id: idx for idx, token_id in enumerate(ids)}
    nodes = []
    for token_id in ids:
        node = dict(nodes_by_id.get(token_id, {}))
        node.setdefault("token_id", token_id)
        node.setdefault("token_raw", str(token_id))
        node.setdefault("token_display", "")
        node["connected_count"] = int(node.get("connected_count") or degree.get(token_id, 0))
        node.setdefault("magnitude", 0.0)
        nodes.append(node)
    edges = [
        {"source": index[source], "target": index[target], "source_token_id": source, "target_token_id": target, "angle_deg": round(angle, 6)}
        for (source, target), angle in sorted(edges_by_pair.items())
    ]
    positions = force_layout(nodes, edges)
    return {
        "meta": {
            "renderer": "canvas2d-precompiled-csv",
            "source_adjacency_csv": str(adjacency_csv),
            "source_token_list_csv": "" if token_list_csv is None else str(token_list_csv),
            "source_dictionary_json": "" if dictionary_json is None else str(dictionary_json),
            "nodes": len(nodes),
            "edges": len(edges),
            "frames": 1,
        },
        "nodes": nodes,
        "edges": edges,
        "frames": [{"xy": positions}],
    }


def force_layout(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[float]]:
    count = len(nodes)
    width = max(1040, min(3200, 900 + count * 18))
    height = max(720, min(2400, 680 + count * 9))
    radius = 0 if count == 1 else max(180, min(width, height) / 2 - 115)
    sim = []
    for i in range(count):
        angle = -math.pi / 2 + (2 * math.pi * i) / max(1, count)
        sim.append({"x": width / 2 + radius * math.cos(angle), "y": height / 2 + radius * math.sin(angle), "vx": 0.0, "vy": 0.0})
    if count <= 1:
        return [[sim[0]["x"], sim[0]["y"]]] if sim else []
    if count > 400:
        # Large CSV exports are where DOM renderers struggle most. Use the
        # deterministic circular placement to keep precompilation quick; users
        # can still drag nodes in the viewer for local inspection.
        return [[round(node["x"], 3), round(node["y"], 3)] for node in sim]
    ideal_link = max(120, min(260, 110 + count * 2.0))
    repel = max(2600, min(11000, 5200 + count * 36))
    iterations = max(80, min(240, 80 + count * 2))
    sim_edges = [(int(e["source"]), int(e["target"])) for e in edges]
    for step in range(iterations):
        alpha = 1.0 - step / iterations
        for i in range(count):
            a = sim[i]
            for j in range(i + 1, count):
                b = sim[j]
                dx = b["x"] - a["x"]
                dy = b["y"] - a["y"]
                distance2 = dx * dx + dy * dy
                if distance2 < 0.01:
                    dx, dy, distance2 = 0.1, -0.1, 0.02
                distance = math.sqrt(distance2)
                force = (repel * alpha) / max(80, distance2)
                fx = (dx / distance) * force
                fy = (dy / distance) * force
                a["vx"] -= fx; a["vy"] -= fy; b["vx"] += fx; b["vy"] += fy
        for si, ti in sim_edges:
            a, b = sim[si], sim[ti]
            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]
            distance = max(1.0, math.sqrt(dx * dx + dy * dy))
            force = (distance - ideal_link) * 0.012 * alpha
            fx = (dx / distance) * force
            fy = (dy / distance) * force
            a["vx"] += fx; a["vy"] += fy; b["vx"] -= fx; b["vy"] -= fy
        for node in sim:
            node["vx"] += (width / 2 - node["x"]) * 0.0025 * alpha
            node["vy"] += (height / 2 - node["y"]) * 0.0025 * alpha
            node["vx"] *= 0.78; node["vy"] *= 0.78
            node["x"] = max(150, min(width - 150, node["x"] + node["vx"]))
            node["y"] = max(54, min(height - 54, node["y"] + node["vy"]))
    return [[round(node["x"], 3), round(node["y"], 3)] for node in sim]


def make_graph(nodes: int, edges_per_node: int, frames: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    ids = list(range(nodes))
    node_rows = [{"token_id": i, "token_raw": f"token_{i}", "token_display": f"token_{i}", "connected_count": 0, "magnitude": 0.0} for i in ids]
    edge_map: dict[tuple[int, int], float] = {}
    for i in range(nodes):
        for k in range(1, edges_per_node + 1):
            edge_map[tuple(sorted((i, (i + k) % nodes)))] = rng.uniform(8.0, 45.0)
        j = rng.randrange(nodes)
        if i != j:
            edge_map[tuple(sorted((i, j)))] = rng.uniform(20.0, 75.0)
    edges = [{"source": s, "target": t, "source_token_id": s, "target_token_id": t, "angle_deg": round(a, 6)} for (s, t), a in sorted(edge_map.items())]
    for e in edges:
        node_rows[e["source"]]["connected_count"] += 1
        node_rows[e["target"]]["connected_count"] += 1
    base = force_layout(node_rows, edges)
    graph_frames = []
    for f in range(frames):
        phase = 2 * math.pi * f / max(1, frames)
        graph_frames.append({"xy": [[round(x + 8 * math.sin(phase + i * 0.17), 3), round(y + 8 * math.cos(phase + i * 0.13), 3)] for i, (x, y) in enumerate(base)]})
    return {
        "meta": {"renderer": "canvas2d-demo", "nodes": nodes, "edges": len(edges), "frames": frames},
        "nodes": node_rows,
        "edges": edges,
        "frames": graph_frames,
    }


def load_graph(path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    if path:
        with path.open() as f:
            return json.load(f)
    if args.adjacency_csv:
        return graph_from_adjacency_csv(args.adjacency_csv, args.token_list_csv, args.dictionary_json)
    return make_graph(args.nodes, args.edges_per_node, args.frames, args.seed)

def render_frame(task: tuple[int, dict[str, Any], str]) -> str:
    index, graph, out_dir = task
    coords = graph["frames"][index]["xy"]
    xs = [point[0] for point in coords] or [0.0]
    ys = [point[1] for point in coords] or [0.0]
    margin = 180
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    width = max(1000, int(max_x - min_x))
    height = max(720, int(max_y - min_y))
    def sx(x: float) -> float:
        return x - min_x
    def sy(y: float) -> float:
        return y - min_y
    min_by_node: dict[int, float] = {}
    for edge in graph["edges"]:
        for key in (int(edge["source"]), int(edge["target"])):
            min_by_node[key] = min(min_by_node.get(key, float("inf")), float(edge["angle_deg"]))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#020617"/>',
    ]
    for edge in graph["edges"]:
        x1, y1 = coords[edge["source"]]
        x2, y2 = coords[edge["target"]]
        is_min = edge["angle_deg"] <= min_by_node[int(edge["source"])] + 1e-9 or edge["angle_deg"] <= min_by_node[int(edge["target"])] + 1e-9
        color = "#fbbf24" if is_min else "#94a3b8"
        opacity = "0.98" if is_min else "0.52"
        stroke_width = "3.2" if is_min else "1.25"
        parts.append(f'<line x1="{sx(x1):.2f}" y1="{sy(y1):.2f}" x2="{sx(x2):.2f}" y2="{sy(y2):.2f}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="{stroke_width}"/>')
        parts.append(f'<text x="{(sx(x1)+sx(x2))/2:.2f}" y="{(sy(y1)+sy(y2))/2:.2f}" fill="{color}" font-size="10" font-family="monospace">{float(edge["angle_deg"]):.2f}°</text>')
    for node, (x, y) in zip(graph["nodes"], coords):
        px, py = sx(x) - 75, sy(y) - 27
        token = html.escape(str(node.get("token_raw", node.get("label", node.get("token_id", ""))))[:22])
        token_id = html.escape(str(node.get("token_id", "")))
        parts.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="150" height="54" rx="12" fill="#111827" stroke="#38bdf8" stroke-width="1.1"/>')
        parts.append(f'<text x="{px+75:.2f}" y="{py+20:.2f}" fill="#f8fafc" font-size="13" font-weight="700" text-anchor="middle" font-family="monospace">{token_id}</text>')
        parts.append(f'<text x="{px+75:.2f}" y="{py+38:.2f}" fill="#cbd5e1" font-size="11" text-anchor="middle" font-family="monospace">{token}</text>')
    parts.append("</svg>")
    out = Path(out_dir) / f"frame_{index:05d}.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    return str(out)

def maybe_render_video(graph: dict[str, Any], frame_dir: Path | None, video_out: Path | None, workers: int) -> str:
    if not frame_dir:
        return ""
    frame_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(render_frame, [(i, graph, str(frame_dir)) for i in range(len(graph["frames"]))]))
    if video_out:
        cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", str(frame_dir / "frame_%05d.svg"), "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", str(video_out)]
        subprocess.run(cmd, check=True)
        return f'<h3>Pre-rendered video</h3><video controls loop src="{video_out.name}" style="width:100%"></video>'
    return "<p class=\"muted\">Pre-rendered SVG frames are available next to this HTML.</p>"

def main() -> None:
    parser = argparse.ArgumentParser(description="Precompile recursive min-angle graph CSV exports into a fast canvas HTML viewer.")
    parser.add_argument("--input-json", type=Path, help="Optional already-compiled graph JSON with nodes, edges, and frames arrays.")
    parser.add_argument("--adjacency-csv", type=Path, help="Existing recursive_group_*_adjacency.csv export to precompile.")
    parser.add_argument("--token-list-csv", type=Path, help="Optional recursive_group_token_list.csv export for token labels/metadata.")
    parser.add_argument("--dictionary-json", type=Path, help="Optional recursive_group_*_dictionary.json export for token labels/metadata.")
    parser.add_argument("--output-html", type=Path, default=Path("out/min_angle_graph_fast_viewer.html"))
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--edges-per-node", type=int, default=4)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frame-dir", type=Path, help="Optionally render SVG frames concurrently for video/offline playback.")
    parser.add_argument("--video-out", type=Path, help="Optional WebM path assembled from --frame-dir with ffmpeg.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    graph = load_graph(args.input_json, args)
    video_block = maybe_render_video(graph, args.frame_dir, args.video_out, args.workers)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", json.dumps(graph, separators=(",", ":"))).replace("__VIDEO_BLOCK__", video_block)
    args.output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output_html} with {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, {len(graph['frames'])} frames")

if __name__ == "__main__":
    main()
