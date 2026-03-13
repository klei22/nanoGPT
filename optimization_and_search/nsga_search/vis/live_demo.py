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
import json, os


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


def create_interactive_generational_scatter(
    file_name_base: str,
    output_path: str = "htmls/interactive_generational_scatter.html",
    start_gen: int = 0,
    end_gen: int = 10,
    server_port: int = 8000,
):
    
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

        # Extract objective/aux arrays for this generation
        val_loss_vals = [eva.objs[0] for eva in population.evaluations]
        energy_vals = [eva.objs[1] for eva in population.evaluations]
        ttft_vals = [eva.objs[2] for eva in population.evaluations]
        size_vals = [eva.aux.get('params', np.nan) for eva in population.evaluations]

        # Ensure all lists have the same length
        min_len = min(
            len(val_loss_vals),
            len(energy_vals),
            len(ttft_vals),
            len(raw_inds) if isinstance(raw_inds, list) and len(raw_inds) > 0 else 10**9,
        )

        for i in range(min_len):
            cfg_dict = raw_inds[i] if isinstance(raw_inds, list) and i < len(raw_inds) else {}
            pop_data.append({
                'generation': gen,
                'validation_loss': val_loss_vals[i],
                'energy_per_token': energy_vals[i],
                'ttft': ttft_vals[i],
                'size': size_vals[i],
                'individual_id': i,
                'config': cfg_dict
            })

    df_pop = pd.DataFrame(pop_data)
    # Prepare formatted config strings for embedding into customdata
    if 'config' in df_pop.columns:
        df_pop['config_str'] = df_pop['config'].apply(_format_cfg_block)
    os.makedirs("logs", exist_ok=True)
    df_pop.to_csv("logs/interactive_scatter_population_data.csv", index=False)

    # Create a single 2D plot: Size vs Validation Loss
    fig_2d = create_size_vs_val_plot(df_pop, generations)

    # Save file and immediately serve locally
    out2d = output_path.replace('.html', '_2d.html')
    _write_2d_html_with_details(fig_2d, out2d, div_id="gen2d")
    print(f"✅ Interactive 2D scatter plot (Size vs Val Loss) saved to: {out2d}")
    try:
        _serve_and_open(out2d, port=server_port)
    except Exception as e:
        print(f"⚠️ Could not auto-launch local server: {e}")
    
    return


def create_size_vs_val_plot(df, generations):
    """Create a single 2D scatter (Size vs Validation Loss) with generation slider highlighting."""
    fig = go.Figure()

    highlight_color = 'red'
    faded_color = 'lightgray'

    gens_present = []
    bg_idx = {}
    hl_idx = {}

    # Background traces (always visible)
    for gen in generations:
        gen_data = df[df['generation'] == gen]
        if gen_data.empty:
            continue
        gens_present.append(gen)
        if 'config_str' in gen_data.columns and 'config' in gen_data.columns:
            customdata_bg = [[cfg_text, cfg_obj] for cfg_text, cfg_obj in zip(gen_data['config_str'].tolist(), gen_data['config'].tolist())]
        elif 'config_str' in gen_data.columns:
            customdata_bg = [[cfg_text, None] for cfg_text in gen_data['config_str'].tolist()]
        else:
            customdata_bg = [["{}", None]] * len(gen_data)
        fig.add_trace(
            go.Scatter(
                x=gen_data['size'],
                y=gen_data['validation_loss'],
                mode='markers',
                marker=dict(size=8, color=faded_color, opacity=0.3, line=dict(width=1, color='gray')),
                name=f'Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' + 'Size: %{x:.3f}<br>' + 'Validation Loss: %{y:.3f}<extra></extra>',
                customdata=customdata_bg,
                visible=True,
                showlegend=False,
            )
        )
        bg_idx[gen] = [len(fig.data) - 1]

    # Highlighted traces (slider controls which gen is shown)
    for i, gen in enumerate(generations):
        gen_data = df[df['generation'] == gen]
        if gen_data.empty:
            continue
        visible = (len(hl_idx) == 0)  # only the first present gen visible initially
        if 'config_str' in gen_data.columns and 'config' in gen_data.columns:
            customdata_hl = [[cfg_text, cfg_obj] for cfg_text, cfg_obj in zip(gen_data['config_str'].tolist(), gen_data['config'].tolist())]
        elif 'config_str' in gen_data.columns:
            customdata_hl = [[cfg_text, None] for cfg_text in gen_data['config_str'].tolist()]
        else:
            customdata_hl = [["{}", None]] * len(gen_data)
        fig.add_trace(
            go.Scatter(
                x=gen_data['size'],
                y=gen_data['validation_loss'],
                mode='markers',
                marker=dict(size=10, color=highlight_color, opacity=0.85, line=dict(width=2, color='darkred')),
                name=f'Current: Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' + 'Size: %{x:.3f}<br>' + 'Validation Loss: %{y:.3f}<extra></extra>',
                customdata=customdata_hl,
                visible=visible,
                showlegend=True if visible else False,
            )
        )
        hl_idx[gen] = [len(fig.data) - 1]

    # Slider steps
    steps = []
    total_traces = len(fig.data)
    base_visible = [False] * total_traces
    for gen, idxs in bg_idx.items():
        for ii in idxs:
            base_visible[ii] = True
    for gen in gens_present:
        vis = base_visible.copy()
        for ii in hl_idx.get(gen, []):
            vis[ii] = True
        steps.append(dict(method='update', args=[{"visible": vis}], label=str(gen)))

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Current Generation: "}, pad={"t": 40}, steps=steps)],
        title=dict(text="Size vs Validation Loss", x=0.5, font=dict(size=16)),
        height=600,
        width=600,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(title_text="Model Size (params)")
    fig.update_yaxes(title_text="Validation Loss")
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
    candidate_ports = [port, port + 1, port + 2]
    tried = []
    for p in candidate_ports:
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
            tried.append(p)
            continue
    raise RuntimeError(f"Unable to bind a local HTTP port (tried {tried})")


def main():
    """Main function to create the interactive plots"""
    import os
    
    # take arguments from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Interactive Generational Scatter Plots")
    parser.add_argument("--ckpt_base", type=str, default="ckpts/infi_medium/ckpt_gen", help="Path to the evolution log file")
    parser.add_argument("--start_gen", type=int, default=0, help="Starting generation index")
    parser.add_argument("--end_gen", type=int, default=100, help="Ending generation index")
    parser.add_argument("--output", type=str, default="htmls/live_demo.html", help="Output HTML file path")
    parser.add_argument("--port", type=int, default=8000, help="Preferred local server port (falls back to +1/+2 if busy)")
    args = parser.parse_args()
    
    file_name_base = args.ckpt_base
    start_gen = args.start_gen
    end_gen = args.end_gen
    output_path = args.output

    create_interactive_generational_scatter(
        file_name_base,
        output_path,
        start_gen,
        end_gen,
        server_port=args.port,
    )

    print("\n✅ Live demo ready!")
    print("📁 File created:")
    print(f"   {args.output.replace('.html', '_2d.html')} - Size vs Val Loss interactive scatter")
    print("\n🌐 The local server should have opened your browser automatically. If not, open the file URL shown above.")


if __name__ == "__main__":
    main()
    
# python interactive_generation_scatter.py --ckpt_base="ckpts/infi_attn_exp_2/1009_0627_ckpt_gen" --start_gen=1 --end_gen=30
