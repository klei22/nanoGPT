#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for cfg in "$ROOT"/configs/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  python "$ROOT/scripts/run_isotropic_demo.py" --config "$cfg" --outdir "$ROOT/reports/$name"
done
cat > "$ROOT/reports/index.html" <<'HTML'
<!doctype html><meta charset="utf-8"><title>Isotropic Random-Distractor Reports</title>
<h1>Isotropic Random-Distractor Reports</h1>
<p>Generated exploration reports:</p>
<ul>
  <li><a href="base_demo/index.html">base_demo</a></li>
  <li><a href="large_vocab/index.html">large_vocab</a></li>
  <li><a href="stress_mean_field/index.html">stress_mean_field</a></li>
</ul>
HTML
echo "$ROOT/reports/index.html"
