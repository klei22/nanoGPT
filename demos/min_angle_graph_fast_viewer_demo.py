#!/usr/bin/env python3
"""Generate a fast min-angle graph viewer.

The original Plotly-style approach is expensive for large animated graphs because every
node/edge/label is a DOM/WebGL trace object. This demo keeps the full graph data in
compact JSON arrays, draws the visible frame on one Canvas 2D surface, and optionally
pre-renders frames in parallel for video assembly.
"""
from __future__ import annotations

import argparse
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
<title>Fast Min-Angle Graph Viewer</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
  body { margin: 0; background: #020617; color: #e2e8f0; }
  header { padding: 12px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; background: #0f172a; border-bottom: 1px solid #1e293b; }
  button, input { accent-color: #38bdf8; }
  button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 7px 10px; cursor: pointer; }
  button:hover { background: #334155; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; min-height: calc(100vh - 58px); }
  #stage { position: relative; overflow: hidden; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  aside { border-left: 1px solid #1e293b; padding: 14px; background: #0b1120; overflow: auto; }
  .stat { display: grid; grid-template-columns: 1fr auto; gap: 8px; padding: 5px 0; border-bottom: 1px solid #172033; }
  .muted { color: #94a3b8; }
  #tip { position: absolute; pointer-events: none; display: none; max-width: 360px; padding: 8px 10px; border: 1px solid #334155; border-radius: 8px; background: rgba(15,23,42,.94); box-shadow: 0 8px 30px rgba(0,0,0,.4); font-size: 12px; white-space: pre-wrap; }
  .pill { border: 1px solid #334155; border-radius: 999px; padding: 4px 8px; color: #cbd5e1; }
</style>
</head>
<body>
<header>
  <strong>Fast Min-Angle Graph Viewer</strong>
  <button id="play">Pause</button>
  <label>Frame <input id="frame" type="range" min="0" max="0" value="0" /></label>
  <label><input id="labels" type="checkbox" /> labels</label>
  <label><input id="minOnly" type="checkbox" /> min-angle edges only</label>
  <span id="readout" class="pill"></span>
</header>
<main>
  <section id="stage"><canvas id="canvas"></canvas><div id="tip"></div></section>
  <aside>
    <h2>Full information, faster rendering</h2>
    <p class="muted">All nodes, edges, angles, tokens, and per-frame coordinates remain available. The speedup comes from drawing into a single canvas instead of maintaining thousands of Plotly/SVG objects per frame.</p>
    <div id="stats"></div>
    __VIDEO_BLOCK__
    <h3>Hover detail</h3>
    <pre id="detail" class="muted">Move the mouse over a node.</pre>
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
frameSlider.max = String(graph.frames.length - 1);
let playing = true, frame = 0, zoom = 1, panX = 0, panY = 0, drag = null, hover = -1;
const minAngle = Math.min(...graph.edges.map(e => e.angle_deg));
const minCutoff = minAngle + graph.meta.min_edge_epsilon_deg;
function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const r = stage.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(r.width * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
}
function project(x, y) {
  const r = stage.getBoundingClientRect();
  return [r.width * .5 + (x * r.width * .44 + panX) * zoom, r.height * .5 + (y * r.height * .44 + panY) * zoom];
}
function unproject(px, py) {
  const r = stage.getBoundingClientRect();
  return [((px - r.width * .5) / zoom - panX) / (r.width * .44), ((py - r.height * .5) / zoom - panY) / (r.height * .44)];
}
function draw() {
  const r = stage.getBoundingClientRect();
  ctx.fillStyle = '#020617'; ctx.fillRect(0, 0, r.width, r.height);
  const coords = graph.frames[frame].xy;
  const minOnly = document.getElementById('minOnly').checked;
  ctx.lineCap = 'round';
  for (const e of graph.edges) {
    const isMin = e.angle_deg <= minCutoff;
    if (minOnly && !isMin) continue;
    const [x1,y1] = project(coords[e.source][0], coords[e.source][1]);
    const [x2,y2] = project(coords[e.target][0], coords[e.target][1]);
    ctx.strokeStyle = isMin ? 'rgba(251,191,36,.95)' : 'rgba(148,163,184,.22)';
    ctx.lineWidth = isMin ? 2.4 : 0.8;
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }
  for (let i = 0; i < graph.nodes.length; i++) {
    const n = graph.nodes[i]; const [x,y] = project(coords[i][0], coords[i][1]);
    ctx.fillStyle = i === hover ? '#fbbf24' : n.color;
    ctx.beginPath(); ctx.arc(x, y, i === hover ? 5.5 : 3.5, 0, Math.PI*2); ctx.fill();
    if (document.getElementById('labels').checked || i === hover) {
      ctx.fillStyle = '#e2e8f0'; ctx.font = '11px ui-monospace, monospace'; ctx.fillText(n.label, x + 7, y - 7);
    }
  }
  readout.textContent = `frame ${frame + 1}/${graph.frames.length} · ${graph.nodes.length} nodes · ${graph.edges.length} edges`;
}
function nearestNode(ev) {
  const rect = canvas.getBoundingClientRect(); const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  const coords = graph.frames[frame].xy; let best = -1, bestD = 12 / zoom;
  const [ux, uy] = unproject(px, py);
  for (let i = 0; i < coords.length; i++) {
    const d = Math.hypot(coords[i][0] - ux, coords[i][1] - uy);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
canvas.addEventListener('mousemove', ev => {
  if (drag) { panX += (ev.clientX - drag.x) / zoom; panY += (ev.clientY - drag.y) / zoom; drag = {x: ev.clientX, y: ev.clientY}; draw(); return; }
  hover = nearestNode(ev); draw();
  if (hover >= 0) {
    const n = graph.nodes[hover];
    const text = JSON.stringify(n, null, 2);
    detail.textContent = text; tip.textContent = text; tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 14) + 'px';
  } else { tip.style.display = 'none'; detail.textContent = 'Move the mouse over a node.'; }
});
canvas.addEventListener('mousedown', ev => { drag = {x: ev.clientX, y: ev.clientY}; });
window.addEventListener('mouseup', () => { drag = null; });
canvas.addEventListener('wheel', ev => { ev.preventDefault(); zoom *= ev.deltaY < 0 ? 1.08 : .925; zoom = Math.max(.2, Math.min(20, zoom)); draw(); }, {passive:false});
frameSlider.addEventListener('input', () => { frame = Number(frameSlider.value); draw(); });
document.getElementById('labels').addEventListener('change', draw);
document.getElementById('minOnly').addEventListener('change', draw);
document.getElementById('play').addEventListener('click', ev => { playing = !playing; ev.target.textContent = playing ? 'Pause' : 'Play'; });
function tick() { if (playing) { frame = (frame + 1) % graph.frames.length; frameSlider.value = String(frame); draw(); } requestAnimationFrame(tick); }
document.getElementById('stats').innerHTML = Object.entries(graph.meta).map(([k,v]) => `<div class="stat"><span>${k}</span><strong>${v}</strong></div>`).join('');
window.addEventListener('resize', resize); resize(); tick();
</script>
</body>
</html>
"""

def make_graph(nodes: int, edges_per_node: int, frames: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    base = []
    for i in range(nodes):
        a = 2 * math.pi * i / nodes
        radius = 0.15 + 0.8 * math.sqrt((i + 0.5) / nodes)
        base.append((radius * math.cos(a), radius * math.sin(a)))
    edge_map: dict[tuple[int, int], float] = {}
    for i in range(nodes):
        for k in range(1, edges_per_node + 1):
            j = (i + k) % nodes
            edge_map[tuple(sorted((i, j)))] = 0.0
        j = rng.randrange(nodes)
        if i != j:
            edge_map[tuple(sorted((i, j)))] = 0.0
    edges = []
    for s, t in edge_map:
        dx = base[s][0] - base[t][0]
        dy = base[s][1] - base[t][1]
        angle = 1.0 + min(179.0, math.hypot(dx, dy) * 90.0)
        edges.append({"source": s, "target": t, "angle_deg": round(angle, 5)})
    min_neighbor = {i: 180.0 for i in range(nodes)}
    for e in edges:
        min_neighbor[e["source"]] = min(min_neighbor[e["source"]], e["angle_deg"])
        min_neighbor[e["target"]] = min(min_neighbor[e["target"]], e["angle_deg"])
    palette = ["#38bdf8", "#818cf8", "#34d399", "#fb7185", "#c084fc"]
    graph_frames = []
    for f in range(frames):
        phase = 2 * math.pi * f / frames
        xy = []
        for i, (x, y) in enumerate(base):
            wobble = 0.025 * math.sin(phase + i * 0.17)
            xy.append([round(x + wobble * math.cos(i), 6), round(y + wobble * math.sin(i * 1.7), 6)])
        graph_frames.append({"xy": xy})
    return {
        "meta": {"renderer": "canvas2d", "nodes": nodes, "edges": len(edges), "frames": frames, "min_edge_epsilon_deg": 0.05},
        "nodes": [{"id": i, "label": f"tok_{i}", "token": f"token_{i}", "min_angle_deg": round(min_neighbor[i], 5), "color": palette[i % len(palette)]} for i in range(nodes)],
        "edges": edges,
        "frames": graph_frames,
    }

def load_graph(path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    if path:
        with path.open() as f:
            return json.load(f)
    return make_graph(args.nodes, args.edges_per_node, args.frames, args.seed)

def render_frame(task: tuple[int, dict[str, Any], str]) -> str:
    index, graph, out_dir = task
    coords = graph["frames"][index]["xy"]
    width = height = 1000
    def sx(x: float) -> float:
        return width * (0.5 + 0.44 * x)
    def sy(y: float) -> float:
        return height * (0.5 + 0.44 * y)
    min_angle = min(e["angle_deg"] for e in graph["edges"])
    cutoff = min_angle + graph["meta"].get("min_edge_epsilon_deg", 0.05)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#020617"/>',
    ]
    for e in graph["edges"]:
        x1, y1 = coords[e["source"]]; x2, y2 = coords[e["target"]]
        is_min = e["angle_deg"] <= cutoff
        color = "#fbbf24" if is_min else "#64748b"
        opacity = "0.9" if is_min else "0.22"
        lw = "2.2" if is_min else "0.6"
        parts.append(f'<line x1="{sx(x1):.2f}" y1="{sy(y1):.2f}" x2="{sx(x2):.2f}" y2="{sy(y2):.2f}" stroke="{color}" stroke-opacity="{opacity}" stroke-width="{lw}"/>')
    for node, (x, y) in zip(graph["nodes"], coords):
        color = node.get("color", "#38bdf8")
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="{color}"/>')
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
    parser = argparse.ArgumentParser(description="Create a fast canvas min-angle graph HTML viewer and optional pre-rendered video frames.")
    parser.add_argument("--input-json", type=Path, help="Optional graph JSON with nodes, edges, and frames arrays.")
    parser.add_argument("--output-html", type=Path, default=Path("out/min_angle_graph_fast_viewer.html"))
    parser.add_argument("--nodes", type=int, default=1500)
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
