#!/usr/bin/env python3
"""Small Flask webapp for LM-head pairwise angle checkpoint comparisons."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_lm_head_pairwise_angles import compare_lm_head_pairwise_angles, plot_html


def find_checkpoints(root: str, limit: int = 500) -> list[str]:
    root_path = Path(root).resolve()
    paths = []
    for path in root_path.rglob("*.pt"):
        if path.is_file():
            paths.append(str(path))
            if len(paths) >= limit:
                break
    return sorted(paths)


def create_app(ckpt_root: str = "."):
    try:
        from flask import Flask, request
    except ImportError as exc:
        raise SystemExit("Flask is required for the webapp. Install it with `pip install flask`.") from exc

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        ckpts = find_checkpoints(ckpt_root)
        default_a = ckpts[0] if ckpts else ""
        default_b = ckpts[1] if len(ckpts) > 1 else default_a
        result_html = ""
        error = ""
        form = request.form if request.method == "POST" else {}
        ckpt_a = form.get("ckpt_a", default_a)
        ckpt_b = form.get("ckpt_b", default_b)
        meta = form.get("meta", "")
        device = form.get("device", "cpu")
        min_angle = float(form.get("min_angle", 0.0) or 0.0)
        max_angle = float(form.get("max_angle", 180.0) or 180.0)
        if request.method == "POST":
            try:
                result = compare_lm_head_pairwise_angles(
                    ckpt_a, ckpt_b, meta=meta or None, device=device,
                    min_angle=min_angle, max_angle=max_angle,
                )
                metrics = "".join(f"<tr><th>{k}</th><td>{v:.6g}</td></tr>" for k, v in result.metrics.items())
                result_html = f"<h2>Metrics</h2><table border='1' cellpadding='4'>{metrics}</table>" + plot_html(result)
            except Exception as exc:  # render user-facing analysis errors in the page
                error = f"<pre style='color:#b00'>{type(exc).__name__}: {exc}</pre>"
        options = "".join(f"<option value='{p}' {'selected' if p == ckpt_a else ''}>{p}</option>" for p in ckpts)
        options_b = "".join(f"<option value='{p}' {'selected' if p == ckpt_b else ''}>{p}</option>" for p in ckpts)
        return f"""<!doctype html><html><head><title>LM head pairwise angles</title>
<style>body{{font-family:sans-serif;margin:2rem}} label{{display:block;margin:.5rem 0}} input,select{{min-width:28rem}}</style></head><body>
<h1>LM head pairwise angle comparison</h1>
<form method='post'>
<label>Checkpoint A <select name='ckpt_a'>{options}</select></label>
<label>Checkpoint B <select name='ckpt_b'>{options_b}</select></label>
<label>Meta pickle for token labels <input name='meta' value='{meta or 'data/shakespeare_char/meta.pkl'}'></label>
<label>Device <select name='device'><option {'selected' if device=='cpu' else ''}>cpu</option><option {'selected' if device=='auto' else ''}>auto</option><option {'selected' if device=='cuda' else ''}>cuda</option></select></label>
<label>Minimum checkpoint-A pair angle in degrees <input type='number' step='0.1' name='min_angle' value='{min_angle}'></label>
<label>Maximum checkpoint-A pair angle in degrees <input type='number' step='0.1' name='max_angle' value='{max_angle}'></label>
<button type='submit'>Compare</button>
</form>{error}{result_html}</body></html>"""

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_app(args.ckpt_root).run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
