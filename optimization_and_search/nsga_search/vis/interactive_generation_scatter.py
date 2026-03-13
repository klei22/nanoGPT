#!/usr/bin/env python3
"""
Interactive Generational Scatter Plot with Plotly

Creates an interactive scatter plot showing evolution across generations with:
- Slider to select current generation (highlighted)
- Other generations shown as faded/shaded points
- Separate 2D plots and 3D plot
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import pandas as pd
from visualization.evolution_history import LogParser
from nsga2 import Population
from typing import Dict, List
import json, os, math


def _pretty_json(d: dict) -> str:
        try:
                return json.dumps(d if isinstance(d, dict) else {}, indent=2, sort_keys=True)
        except Exception:
                return "{}"


def _find_layers_list(cfg: dict):
    """Best-effort: locate the per-layer list of dicts in the config."""
    if not isinstance(cfg, dict):
        return []
    # Prefer explicit nested structure from checkpoints
    if isinstance(cfg.get("layers"), list):
        return cfg.get("layers")
    # common keys
    for k in ("layers", "layer_settings", "layer_cfgs", "per_layer", "blocks"):
        v = cfg.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    # generic scan
    for v in cfg.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            sample = v[0]
            if any(key in sample for key in ("n_head", "mlp_size", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "n_cproj", "attention_variant")):
                return v
    return []


def _extract_globals(cfg: dict) -> dict:
    """Return a dict of global settings regardless of nesting (flat or cfg['globals'])."""
    if not isinstance(cfg, dict):
        return {}
    g = cfg.get('globals') if isinstance(cfg.get('globals'), dict) else cfg
    return {
        'n_embd': g.get('n_embd'),
        'block_size': g.get('block_size'),
        'use_concat_heads': g.get('use_concat_heads'),
        'layer_mask': g.get('layer_mask') or []
    }


def _estimate_params_simple(cfg: dict) -> float:
    """Rough parameter estimate in millions (M). Best-effort and may not match backend exactly."""
    try:
        g = _extract_globals(cfg)
        d = int(g.get("n_embd") or 0)
        layer_mask = g.get("layer_mask") or []
        use_concat = bool(g.get("use_concat_heads", True))
        layers = _find_layers_list(cfg)
        total = 0
        L = max(len(layers), len(layer_mask))
        for i in range(L):
            if layer_mask and (i >= len(layer_mask) or not layer_mask[i]):
                continue
            li = layers[i] if i < len(layers) else {}
            h = int(li.get('n_head', 8) or 8)
            g = int(li.get('n_kv_group', h) or h)
            dq = int(li.get('n_qk_head_dim', max(1, d // max(h, 1))) or max(1, d // max(h, 1)))
            dv = int(li.get('n_v_head_dim', dq) or dq)
            cproj = int(li.get('n_cproj', 1) or 1)
            mlp = int(li.get('mlp_size', 4 * d) or (4 * d))

            # Q, K, V projections
            q_params = d * (h * dq)
            k_params = d * (g * dq)
            v_params = d * (g * dv)
            # Output projection (very rough)
            out_in = (h * dv) if use_concat else dv
            o_params = out_in * d * max(1, cproj)
            # MLP two-linears
            mlp_params = d * mlp + mlp * d

            total += q_params + k_params + v_params + o_params + mlp_params
        return total / 1e6
    except Exception:
        return 0.0


def _format_cfg_block(cfg: dict) -> str:
    """Render the configuration text similar to the run log formatting."""
    try:
        if not isinstance(cfg, dict):
            return _pretty_json({})

        # Globals line (support nested structure: cfg['globals'])
        globals_dict = _extract_globals(cfg)
        globals_line = f"Globals: {str(globals_dict)}"

        # Layers info
        layer_mask = globals_dict.get('layer_mask') or []
        layers = _find_layers_list(cfg)  # cfg['layers'] if present
        total_layers = len(layer_mask) if layer_mask else (len(layers) or 0)
        active_layers = sum(1 for x in layer_mask if x) if layer_mask else (len(layers) or 0)

        header_lines = [
            globals_line,
            f"Total layers: {total_layers}; Active layers: {active_layers}",
        ]

        # Estimated params
        est_m = _estimate_params_simple(cfg)
        if est_m > 0:
            header_lines.append(f"Estimated params: {est_m:.2f}M")

        # Per-layer lines (skip inactive)
        lines = []
        L = max(len(layers), len(layer_mask))
        for i in range(L):
            if layer_mask and (i >= len(layer_mask) or not layer_mask[i]):
                continue
            li = layers[i] if i < len(layers) else {}
            n_head = li.get('n_head', 'NA')
            n_kv_group = li.get('n_kv_group', 'NA')
            mlp_size = li.get('mlp_size', 'NA')
            n_qk_head_dim = li.get('n_qk_head_dim', 'NA')
            n_v_head_dim = li.get('n_v_head_dim', 'NA')
            n_cproj = li.get('n_cproj', 'NA')
            attn_var = li.get('attention_variant', li.get('attn_variant', 'NA'))
            lines.append(
                f"  - Layer {i}: n_head={n_head}, n_kv_group={n_kv_group}, mlp_size={mlp_size}, "
                f"n_qk_head_dim={n_qk_head_dim}, n_v_head_dim={n_v_head_dim}, n_cproj={n_cproj}, attention_variant={attn_var}"
            )

        return "\n".join(header_lines + lines)
    except Exception:
        return _pretty_json(cfg)


def _write_2d_html_with_details(fig: go.Figure, output_path: str, div_id: str = "gen2d"):
    """Write the 2D figure to HTML and add a details panel below that updates on hover."""
    html_template = """<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\"/>
    <title>Interactive Generational Scatter - 2D</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; }}
      .container {{ max-width: 1400px; margin: 0 auto; padding: 16px; }}
      #details-panel {{
        margin-top: 12px;
        padding: 12px;
        border-top: 1px solid #e0e0e0;
        background: #fafafa;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", monospace;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 360px;
      }}
      .details-header {{ font-weight: 600; margin-bottom: 6px; }}
      .details-meta {{ color: #444; margin-bottom: 8px; }}
      .hint {{ color: #666; font-style: italic; }}
    </style>
  </head>
  <body>
    <div class=\"container\">
      {plotly_html}
            <div id=\"gen-label\" style=\"margin: 8px 0 6px 0; font-weight: 600;\"></div>
                    <div id=\"arch-panel\" style=\"margin-top: 10px;\">
                        <div style=\"font-weight:600; margin-bottom:6px;\">Architecture <span id=\"arch-plot-title\" style=\"font-weight:500; margin-left:6px; color:#555;\"></span></div>
                        <div id=\"arch-plot-heads\" style=\"height:180px;\"></div>
                        <div id=\"arch-plot-dims\" style=\"height:180px; margin-top:8px;\"></div>
                        <div id=\"arch-plot-mlp\" style=\"height:180px; margin-top:8px;\"></div>
                    </div>
            <div id=\"details-panel\"> 
                <div class=\"hint\">Click a point to see its full configuration here…</div>
            </div>
    </div>
    {plotly_script}
  </body>
</html>
"""
    post_script = """
(function() {
  var gd = document.getElementById('__DIV_ID__') || document.getElementsByClassName('plotly-graph-div')[0];
  var panel = document.getElementById('details-panel');
    var genLabel = document.getElementById('gen-label');
    if(!gd || !panel) return;
        var lastDetailsHTML = null;
                var lastCfgObj = null;
                var lastArchLabel = '';

  function fmt(val, digits) {
    if (typeof val === 'number' && isFinite(val)) return val.toFixed(digits);
    return String(val);
  }

    function getCurrentGenFromTraces() {
        try {
            var data = gd.data || [];
            for (var i = 0; i < data.length; i++) {
                var tr = data[i];
                if (!tr || !tr.name) continue;
                // Look for the currently visible highlighted trace name: 'Current: Gen X'
                if (tr.name.indexOf('Current: Gen') === 0 && tr.visible !== false && tr.visible !== 'legendonly') {
                    var m = tr.name.match(/Current: Gen\s+(\d+)/);
                    if (m && m[1]) return m[1];
                }
            }
        } catch (e) {}
        return null;
    }

    function updateGenLabel() {
        if (!genLabel) return;
        var g = getCurrentGenFromTraces();
        if (g !== null) {
            genLabel.textContent = 'Generation: ' + g;
        }
    }

            gd.on('plotly_click', function(e) {
    if (!e || !e.points || e.points.length === 0) return;
    var p = e.points[0];

                    var cfgText = null;
                    var cfgObj = null;
                    if (Array.isArray(p.customdata)) {
                        cfgText = p.customdata[0];
                        cfgObj = p.customdata[1] || null;
                    } else {
                        // backward fallback
                        cfgText = p.customdata;
                    }
                if (typeof cfgText !== 'string') {
                    try { cfgText = JSON.stringify(cfgText, null, 2); } catch(err) { cfgText = String(cfgText); }
                }

    var seriesName = p.data && p.data.name ? p.data.name : '';
    var hdr = '<div class="details-header">' + seriesName + '</div>';

    var metaParts = [];
    if (p.text) metaParts.push('Point: ' + p.text);
    if (typeof p.x !== 'undefined') metaParts.push('X=' + fmt(p.x, 5));
    if (typeof p.y !== 'undefined') metaParts.push('Y=' + fmt(p.y, 5));
    var meta = '<div class="details-meta">' + metaParts.join(' | ') + '</div>';

            var content = hdr + meta + '<pre>' + cfgText + '</pre>';
            panel.innerHTML = content;
            lastDetailsHTML = content;
            if (cfgObj) {
                var archLabel = '';
                if (typeof p.text === 'string' && p.text.trim().length) {
                    archLabel = p.text.trim().toLowerCase();
                }
                renderArchPlots(cfgObj, archLabel);
                lastCfgObj = cfgObj;
                lastArchLabel = archLabel;
            }
  });

        function getGlobals(cfg) {
            if (!cfg || typeof cfg !== 'object') return {};
            return (cfg.globals && typeof cfg.globals === 'object') ? cfg.globals : cfg;
        }

        function getLayers(cfg) {
            if (!cfg || typeof cfg !== 'object') return [];
            if (Array.isArray(cfg.layers)) return cfg.layers;
            return [];
        }

        function renderArchPlots(cfg, label) {
            var g = getGlobals(cfg);
            var layers = getLayers(cfg);
            var mask = Array.isArray(g.layer_mask) ? g.layer_mask.slice(0, layers.length) : new Array(layers.length).fill(true);
            var xs = [];
            for (var i = 0; i < layers.length; i++) { if (mask[i]) xs.push(i); }

            var titleEl = document.getElementById('arch-plot-title');
            if (titleEl) {
                titleEl.textContent = label ? ' (' + label + ')' : '';
            }

            var a_heads = [], a_kvg = [], a_qk = [], a_v = [], a_mlp = [], pos = [];
            for (var j = 0; j < xs.length; j++) {
                var li = layers[xs[j]] || {};
                var isInf = (li.attention_variant === 'infinite');
                a_heads.push(isInf ? (li.n_head || 0) : 0);
                a_kvg.push(isInf ? (li.n_kv_group || 0) : 0);
                a_qk.push(isInf ? (li.n_qk_head_dim || 0) : 0);
                a_v.push(isInf ? (li.n_v_head_dim || 0) : 0);
                a_mlp.push(li.mlp_size || 0);
                pos.push(j);
            }

            var ticktext = xs.map(function(i){ return String(i); });
            var tickvals = pos.slice();

            // Plot 1: n_heads vs n_kv_group
            var data1 = [
                {type:'bar', x: tickvals, y: a_heads, name:'n_heads', marker:{color:'#4C72B0'}, offsetgroup:'a'},
                {type:'bar', x: tickvals, y: a_kvg,   name:'n_kv_group', marker:{color:'#DD8452'}, offsetgroup:'b'}
            ];
            var layout1 = {barmode:'group', margin:{l:40,r:10,t:10,b:40}, showlegend:true, legend:{orientation:'h'}, yaxis:{title:'count'}, xaxis:{tickmode:'array', tickvals:tickvals, ticktext:ticktext}};
            Plotly.newPlot('arch-plot-heads', data1, layout1, {displayModeBar:false, responsive:true});

            // Plot 2: qk/v dims
            var data2 = [
                {type:'bar', x: tickvals, y: a_qk, name:'qk_dim', marker:{color:'#C44E52'}, offsetgroup:'a'},
                {type:'bar', x: tickvals, y: a_v,  name:'v_dim',  marker:{color:'#8172B2'}, offsetgroup:'b'}
            ];
            var layout2 = {barmode:'group', margin:{l:40,r:10,t:10,b:40}, showlegend:true, legend:{orientation:'h'}, yaxis:{title:'head dims'}, xaxis:{title:'active layer index', tickmode:'array', tickvals:tickvals, ticktext:ticktext}};
            Plotly.newPlot('arch-plot-dims', data2, layout2, {displayModeBar:false, responsive:true});

            // Plot 3: mlp_size
                var data3 = [{type:'bar', x: tickvals, y: a_mlp, name:'mlp_size', marker:{color:'#55A868'}}];
                var layout3 = {margin:{l:40,r:10,t:10,b:40}, showlegend:true, legend:{orientation:'h'}, yaxis:{title:'mlp_size'}, xaxis:{tickmode:'array', tickvals:tickvals, ticktext:ticktext}};
            Plotly.newPlot('arch-plot-mlp', data3, layout3, {displayModeBar:false, responsive:true});
        }

    function restoreDetails() {
        if (lastDetailsHTML) {
            panel.innerHTML = lastDetailsHTML;
        }
            if (lastCfgObj) {
                renderArchPlots(lastCfgObj, lastArchLabel);
            }
    }
    gd.on('plotly_restyle', restoreDetails);
    gd.on('plotly_relayout', restoreDetails);
    gd.on('plotly_redraw', restoreDetails);
    gd.on('plotly_animated', restoreDetails);
        gd.on('plotly_sliderchange', updateGenLabel);
        gd.on('plotly_restyle', updateGenLabel);
        gd.on('plotly_relayout', updateGenLabel);
        gd.on('plotly_redraw', updateGenLabel);
        gd.on('plotly_animated', updateGenLabel);
        // Initialize label once figure is ready
        if (gd && gd.layout) {
            updateGenLabel();
        } else {
            setTimeout(updateGenLabel, 0);
        }

})();
""".replace('__DIV_ID__', div_id)
    try:
        fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, div_id=div_id)
    except TypeError:
        # Older Plotly versions may not support div_id; fall back without it
        fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    html_str = html_template.format(plotly_html=fig_html, plotly_script=f"<script>{post_script}</script>")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_str)


def create_interactive_generational_scatter(file_name_base: str, output_path: str = "htmls/interactive_generational_scatter.html", start_gen: int = 0, end_gen: int = 10, list_metrics: list = None, port: int = 8000):
    
    pop_data = []
    generations = list(range(start_gen, end_gen + 1))

    for gen in generations:
        json_file_name = f"{file_name_base}{gen}.json"
        if not os.path.exists(json_file_name):
            print(f"❌ Checkpoint file not found: {json_file_name}")
            exit(f"❌ Checkpoint file not found: {json_file_name}\nPlease ensure all generation checkpoint files are present.")

        population = Population.load_checkpoint(json_file_name, from_pkl=False)
        # also load raw JSON to fetch configs
        try:
            with open(json_file_name, "r", encoding="utf-8") as f:
                raw = json.load(f)
            raw_inds = raw.get("individuals", []) or []
        except Exception:
            raw_inds = []

        # Extract objective from auxiliary data of evaluations
        aux_list = [eva.aux for eva in population.evaluations]

        for i in range(len(aux_list)):
            cfg_dict = raw_inds[i] if isinstance(raw_inds, list) and i < len(raw_inds) else {}
            pop_data.append({
                'generation': gen,
                'individual_id': i,
                'config': cfg_dict,
                'aux': aux_list[i] if i < len(aux_list) else {}
            })

    df_pop = pd.DataFrame(pop_data)
    # Prepare formatted config strings for embedding into customdata
    if 'config' in df_pop.columns:
        df_pop['config_str'] = df_pop['config'].apply(_format_cfg_block)
    os.makedirs("logs", exist_ok=True)
    df_pop.to_csv("logs/interactive_scatter_population_data.csv", index=False)

    # Create a single 2D plot: Size vs Validation Loss
    fig_2d = create_2d_plot(df_pop, generations, list_metrics=list_metrics)

    # Save file and immediately serve locally
    out2d = output_path.replace('.html', '_2d.html')
    _write_2d_html_with_details(fig_2d, out2d, div_id="gen2d")
    print(f"✅ Interactive 2D scatter plot saved to: {out2d}")
    try:
        _serve_and_open(out2d, port=port)
    except Exception as e:
        print(f"⚠️ Could not auto-launch local server: {e}")
    
    return


def create_2d_plot(df, generations, list_metrics=None):
    """Create a single 2D scatter (Size vs Validation Loss) with generation slider highlighting."""
    highlight_color = 'red'
    faded_color = 'lightgray'

    if not list_metrics or len(list_metrics) < 2:
        raise ValueError("At least two metrics are required to create scatter plots.")

    # check that all requested metrics exist in aux
    sample_aux = df['aux'].iloc[0] if not df.empty else {}
    for metric in list_metrics:
        if not (isinstance(sample_aux, dict) and metric in sample_aux):
            raise ValueError(f"Metric '{metric}' not found in auxiliary data.")

    # Extract requested metric values from aux dicts
    metric_series = {}
    for metric in list_metrics:
        metric_series[metric] = df['aux'].apply(lambda aux: aux.get(metric) if isinstance(aux, dict) else None)

    metric_pairs = [(list_metrics[i], list_metrics[j]) for i in range(len(list_metrics)) for j in range(i + 1, len(list_metrics))]
    if not metric_pairs:
        raise ValueError("No metric pairs available to plot.")

    num_of_plots = len(metric_pairs)
    cols = min(num_of_plots, max(1, math.ceil(math.sqrt(num_of_plots))))
    rows = math.ceil(num_of_plots / cols)
    subplot_titles = [f"{mx} vs {my}" for mx, my in metric_pairs]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    gens_present: List[int] = []
    bg_idx: Dict[tuple, Dict[int, List[int]]] = {}
    hl_idx: Dict[tuple, Dict[int, List[int]]] = {}
    first_highlight_gen = None

    plot_idx = 0
    for mx, my in metric_pairs:
        plot_idx += 1
        row = math.ceil(plot_idx / cols)
        col = plot_idx - (row - 1) * cols

        for gen in generations:
            mask = df['generation'] == gen
            if not mask.any():
                continue
            gen_frame = df[mask].copy()
            gen_frame['metric_x'] = metric_series[mx][mask]
            gen_frame['metric_y'] = metric_series[my][mask]
            gen_frame = gen_frame.dropna(subset=['metric_x', 'metric_y'])
            if gen_frame.empty:
                continue
            if gen not in gens_present:
                gens_present.append(gen)

            if 'config_str' in gen_frame.columns and 'config' in gen_frame.columns:
                customdata_bg = [[cfg_text, cfg_obj] for cfg_text, cfg_obj in zip(gen_frame['config_str'].tolist(), gen_frame['config'].tolist())]
            elif 'config_str' in gen_frame.columns:
                customdata_bg = [[cfg_text, None] for cfg_text in gen_frame['config_str'].tolist()]
            else:
                customdata_bg = [["{}", None]] * len(gen_frame)

            fig.add_trace(
                go.Scatter(
                    x=gen_frame['metric_x'],
                    y=gen_frame['metric_y'],
                    mode='markers',
                    marker=dict(size=7, color=faded_color, opacity=0.3, line=dict(width=1, color='gray')),
                    name=f'Gen {gen}',
                    text=[f'Gen {gen}, Individual {i}' for i in gen_frame['individual_id']],
                    hovertemplate='<b>%{text}</b><br>' + f'{mx}: %{{x:.3f}}<br>' + f'{my}: %{{y:.3f}}<extra></extra>',
                    customdata=customdata_bg,
                    visible=True,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            bg_idx.setdefault((row, col), {}).setdefault(gen, []).append(len(fig.data) - 1)

        for gen in generations:
            mask = df['generation'] == gen
            if not mask.any():
                continue
            gen_frame = df[mask].copy()
            gen_frame['metric_x'] = metric_series[mx][mask]
            gen_frame['metric_y'] = metric_series[my][mask]
            gen_frame = gen_frame.dropna(subset=['metric_x', 'metric_y'])
            if gen_frame.empty:
                continue

            if 'config_str' in gen_frame.columns and 'config' in gen_frame.columns:
                customdata_hl = [[cfg_text, cfg_obj] for cfg_text, cfg_obj in zip(gen_frame['config_str'].tolist(), gen_frame['config'].tolist())]
            elif 'config_str' in gen_frame.columns:
                customdata_hl = [[cfg_text, None] for cfg_text in gen_frame['config_str'].tolist()]
            else:
                customdata_hl = [["{}", None]] * len(gen_frame)

            if first_highlight_gen is None:
                first_highlight_gen = gen
            visible = (gen == first_highlight_gen)

            fig.add_trace(
                go.Scatter(
                    x=gen_frame['metric_x'],
                    y=gen_frame['metric_y'],
                    mode='markers',
                    marker=dict(size=9, color=highlight_color, line=dict(width=1.5, color='darkred'), opacity=0.85),
                    name=f'Current: Gen {gen}',
                    text=[f'Gen {gen}, Individual {i}' for i in gen_frame['individual_id']],
                    hovertemplate='<b>%{text}</b><br>' + f'{mx}: %{{x:.3f}}<br>' + f'{my}: %{{y:.3f}}<extra></extra>',
                    customdata=customdata_hl,
                    visible=visible,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            hl_idx.setdefault((row, col), {}).setdefault(gen, []).append(len(fig.data) - 1)

    if not gens_present:
        raise ValueError("No generation data available to plot the requested metrics.")

    steps = []
    total_traces = len(fig.data)
    base_visible = [False] * total_traces
    for gen_dict in bg_idx.values():
        for idxs in gen_dict.values():
            for idx in idxs:
                base_visible[idx] = True

    for gen in gens_present:
        vis = base_visible.copy()
        for gen_dict in hl_idx.values():
            for idx in gen_dict.get(gen, []):
                vis[idx] = True
        steps.append(dict(method='update', args=[{"visible": vis}], label=str(gen)))

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Current Generation: "}, pad={"t": 30}, steps=steps)],
        height=max(400, rows * 320),
        width=max(500, cols * 380),
        showlegend=False,
    )

    # apply axis labels per subplot
    plot_idx = 0
    for mx, my in metric_pairs:
        plot_idx += 1
        row = math.ceil(plot_idx / cols)
        col = plot_idx - (row - 1) * cols
        axis_suffix = '' if (row, col) == (1, 1) else str(plot_idx)
        fig.update_xaxes(title_text=mx, row=row, col=col)
        fig.update_yaxes(title_text=my, row=row, col=col)

    return fig

def _serve_and_open(html_path: str, port: int = 8000):
    """Serve the directory containing html_path on localhost, open the page, and keep serving until interrupted."""
    import http.server, socketserver, webbrowser, functools
    directory = os.path.abspath(os.path.dirname(html_path))
    filename = os.path.basename(html_path)

    def try_server(p):
        Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
        httpd = socketserver.TCPServer(("", p), Handler)
        return httpd

    httpd = None
    tried_ports = []
    for offset in range(3):
        p = port + offset
        tried_ports.append(p)
        try:
            httpd = try_server(p)
            url = f"http://localhost:{p}/{filename}"
            print(f"🌐 Serving {directory} at {url}")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            print("Press Ctrl+C to stop the server…")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                httpd.server_close()
            return
        except OSError:
            continue
    raise RuntimeError(f"Unable to bind a local HTTP port (tried {', '.join(str(tp) for tp in tried_ports)})")


def main():
    """Main function to create the interactive plots"""
    import os
    
    # take arguments from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Interactive Generational Scatter Plots")
    parser.add_argument("--ckpt_base", type=str, default="ckpts/infi_hw_medium_corrected/ckpt_gen", help="Path to the evolution log file")
    # parser.add_argument("--ckpt_base", type=str, default="ckpts/infi_hw_med_continue/1117_0732_ckpt_gen", help="Path to the evolution log file")
    parser.add_argument("--start_gen", type=int, default=1, help="Starting generation index")
    parser.add_argument("--end_gen", type=int, default=60, help="Ending generation index")
    parser.add_argument("--metrics", type=str, nargs='+', default=["params", "val_loss", "energy_per_token_uJ", "token_delay"], help="List of metrics to plot")
    # parser.add_argument("--metrics", type=str, nargs='+', default=["params", "val_loss", "kv_cache_size"], help="List of metrics to plot")
    parser.add_argument("--output", type=str, default="htmls/interactive_generational_scatter.html", help="Output HTML file path")
    parser.add_argument("--port", type=int, default=8002, help="Preferred local port for serving the interactive plot")
    args = parser.parse_args()
    
    file_name_base = args.ckpt_base
    start_gen = args.start_gen
    end_gen = args.end_gen
    output_path = args.output

    create_interactive_generational_scatter(file_name_base, output_path, start_gen, end_gen, list_metrics=args.metrics, port=args.port)

    print("\n✅ Live demo ready!")
    print("📁 File created:")
    print(f"   {args.output.replace('.html', '_2d.html')} - interactive scatter")
    print("\n🌐 The local server should have opened your browser automatically. If not, open the file URL shown above.")


if __name__ == "__main__":
    main()
    
# python interactive_generation_scatter.py --ckpt_base="ckpts/infi_attn_exp_2/1009_0627_ckpt_gen" --start_gen=1 --end_gen=30
