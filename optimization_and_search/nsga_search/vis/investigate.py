import numpy as np
import pandas as pd
from nsga2 import Population
import json, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def _limit_colormap_lightness(colors: np.ndarray, max_lightness: float = 0.82) -> np.ndarray:
    """Clamp overly bright colors to avoid near-white regions in plots."""

    if colors.ndim != 2 or colors.shape[1] < 3:
        return colors
    rgb = colors[:, :3]
    # Perceptual luminance weights for RGB (sRGB)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=float)
    lightness = rgb @ weights
    mask = lightness > max_lightness
    if np.any(mask):
        scale = (max_lightness / lightness[mask]).reshape(-1, 1)
        rgb[mask] = np.clip(rgb[mask] * scale, 0.0, 1.0)
        colors[:, :3] = rgb
    return colors



def get_pareto_front(obj1, obj2):
    """Identify the Pareto front from a set of points."""
    points = np.array(list(zip(obj1, obj2)))
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(points[is_pareto] < point, axis=1)
            is_pareto[i] = True
    return points[is_pareto]


def get_active_layers(cfg):
    globals_cfg = cfg.get("globals", {}) if isinstance(cfg, dict) else {}
    mask = globals_cfg.get("layer_mask")
    layers = cfg.get("layers", []) if isinstance(cfg, dict) else []
    if not isinstance(mask, list):
        return layers
    return [layer for layer, active in zip(layers, mask) if active]


def _estimate_layer_params(cfg: dict) -> list:
    """Best-effort estimate of per-layer parameter counts (active layers only)."""

    if not isinstance(cfg, dict):
        return []

    globals_cfg = cfg.get("globals", cfg)
    layers = cfg.get("layers", []) if isinstance(cfg.get("layers"), list) else []
    mask = globals_cfg.get("layer_mask")

    d_model = globals_cfg.get("n_embd") or globals_cfg.get("d_model")
    if not isinstance(d_model, (int, float)) or d_model <= 0:
        return []
    d = int(d_model)

    use_concat = bool(globals_cfg.get("use_concat_heads", True))

    indices = list(range(len(layers)))
    if isinstance(mask, list) and mask:
        indices = [idx for idx, active in enumerate(mask) if active and idx < len(layers)]

    per_layer = []
    for idx in indices:
        layer = layers[idx] if idx < len(layers) else {}
        if not isinstance(layer, dict):
            per_layer.append(0.0)
            continue

        h = int(layer.get("n_head", 8) or 8)
        g_groups = int(layer.get("n_kv_group", h) or h)
        qk_dim = int(layer.get("n_qk_head_dim", max(1, d // max(h, 1))) or max(1, d // max(h, 1)))
        v_dim = int(layer.get("n_v_head_dim", qk_dim) or qk_dim)
        cproj = int(layer.get("n_cproj", 1) or 1)
        mlp_size = int(layer.get("mlp_size", 4 * d) or (4 * d))
        attn_variant = layer.get("attention_variant", layer.get("attn_variant", "mha"))

        # Attention params (rough)
        attn_params = 0
        if attn_variant == "infinite":
            q_params = d * (h * qk_dim)
            k_params = d * (g_groups * qk_dim)
            v_params = d * (g_groups * v_dim)
            if use_concat:
                out_proj = (h * v_dim) * d
            else:
                out_proj = cproj * (v_dim * d)
            attn_params = q_params + k_params + v_params + out_proj
        elif attn_variant == "identity":
            attn_params = 0
        else:  # fallback to multi-head style
            qkv_params = d * (h * (qk_dim + qk_dim + v_dim))
            if use_concat:
                out_proj = (h * v_dim) * d
            else:
                out_proj = cproj * (v_dim * d)
            attn_params = qkv_params + out_proj

        mlp_params = 2 * d * mlp_size
        per_layer.append(float(attn_params + mlp_params))

    return per_layer


def _compute_frontness(weights: np.ndarray) -> tuple:
    if weights.size == 0:
        return 0.0, 0.0
    total = weights.sum()
    if total <= 0:
        return 0.0, 0.0
    if weights.size == 1:
        com = 0.0
    else:
        depth_positions = np.linspace(0.0, 1.0, weights.size)
        com = float((weights * depth_positions).sum() / total)
    frontness = 1.0 - 2.0 * com
    return float(com), float(frontness)


def _compute_layer_stats(cfg: dict) -> dict:
    layers = get_active_layers(cfg)
    heads = []
    mlps = []
    kv_groups = []
    gqa_flags = []
    head_group_ratios = []
    identity_flags = []
    qk_v_ratios = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        attn_variant = layer.get("attention_variant", layer.get("attn_variant"))
        m = layer.get("mlp_size")

        if isinstance(m, (int, float)):
            mlps.append(float(m))

        attn_active = attn_variant != "identity"
        identity_flags.append(0.0 if attn_active else 1.0)
        h = layer.get("n_head") if attn_active else None
        kv = layer.get("n_kv_group") if attn_active else None
        qk_dim = layer.get("n_qk_head_dim") if attn_active else None
        v_dim = layer.get("n_v_head_dim") if attn_active else None

        if isinstance(h, (int, float)):
            heads.append(float(h))
        if isinstance(kv, (int, float)):
            kv_val = float(kv)
            kv_groups.append(kv_val)
            if isinstance(h, (int, float)) and kv_val > 0:
                head_group_ratios.append(float(h) / kv_val)
                gqa_flags.append(1.0 if float(h) > kv_val else 0.0)
            else:
                gqa_flags.append(0.0)
        elif attn_active:
            gqa_flags.append(0.0)

        if attn_active and isinstance(qk_dim, (int, float)) and isinstance(v_dim, (int, float)):
            qk = float(qk_dim)
            v = float(v_dim)
            if v > 0.0:
                qk_v_ratios.append(qk*h / (v*kv))
    head_arr = np.array(heads, dtype=float) if heads else np.array([])
    mlp_arr = np.array(mlps, dtype=float) if mlps else np.array([])
    _, head_front = _compute_frontness(head_arr) if head_arr.size else (0.0, 0.0)
    _, mlp_front = _compute_frontness(mlp_arr) if mlp_arr.size else (0.0, 0.0)
    gqa_arr = np.array(gqa_flags, dtype=float) if gqa_flags else np.array([])
    _, gqa_front = _compute_frontness(gqa_arr) if gqa_arr.size and gqa_arr.sum() > 0 else (0.0, 0.0)
    gqa_fraction = float(gqa_arr.mean()) if gqa_arr.size else 0.0
    identity_arr = np.array(identity_flags, dtype=float) if identity_flags else np.array([])
    identity_fraction = float(identity_arr.mean()) if identity_arr.size else 0.0
    _, identity_front = _compute_frontness(identity_arr) if identity_arr.size and identity_arr.sum() > 0 else (0.0, 0.0)
    qk_v_arr = np.array(qk_v_ratios, dtype=float) if qk_v_ratios else np.array([])
    _, qk_v_front = _compute_frontness(qk_v_arr) if qk_v_arr.size else (0.0, 0.0)

    return {
        "head_counts": heads,
        "mlp_sizes": mlps,
    "kv_groups": kv_groups,
    "head_group_ratios": head_group_ratios,
        "head_frontness": head_front,
        "mlp_frontness": mlp_front,
        "gqa_frontness": gqa_front,
        "gqa_fraction": gqa_fraction,
        "identity_fraction": identity_fraction,
        "identity_frontness": identity_front,
        "qk_v_ratio_distribution": qk_v_ratios,
        "qk_v_frontness": qk_v_front,
    }

def analyze_individual(individual):
    """Extract relevant metrics from an individual for analysis."""

    cfg = individual if isinstance(individual, dict) else getattr(individual, "cfg", {}).get("config", {})
    if isinstance(cfg, dict) and "config" in cfg and isinstance(cfg["config"], dict):
        cfg = cfg["config"]
    if not isinstance(cfg, dict):
        return {}

    globals_cfg = cfg.get("globals", cfg)
    layers = cfg.get("layers", [])
    active_layers = get_active_layers(cfg)

    layer_stats = _compute_layer_stats(cfg)
    head_counts = layer_stats["head_counts"]

    head_arr = np.array(head_counts, dtype=float) if head_counts else None
    stats = {}
    if head_arr is not None and head_arr.size:
        stats = {
            "n_heads_mean": float(np.mean(head_arr)),
            "n_heads_std": float(np.std(head_arr, ddof=0)) if head_arr.size > 1 else 0.0,
            "n_heads_min": float(np.min(head_arr)),
            "n_heads_max": float(np.max(head_arr)),
            "n_heads_median": float(np.median(head_arr)),
            "n_heads_p10": float(np.percentile(head_arr, 10)),
            "n_heads_p90": float(np.percentile(head_arr, 90)),
            "n_heads_sum": float(np.sum(head_arr)),
        }
    else:
        stats = {
            "n_heads_mean": 0.0,
            "n_heads_std": 0.0,
            "n_heads_min": 0.0,
            "n_heads_max": 0.0,
            "n_heads_median": 0.0,
            "n_heads_p10": 0.0,
            "n_heads_p90": 0.0,
            "n_heads_sum": 0.0,
        }

    layer_param_estimates = _estimate_layer_params(cfg)
    layer_param_arr = np.array(layer_param_estimates, dtype=float) if layer_param_estimates else np.array([])
    param_com, param_frontness = _compute_frontness(layer_param_arr) if layer_param_arr.size else (0.0, 0.0)

    return {
        "n_layers_total": len(layers),
        "n_layers_active": len(active_layers),
        **stats,
        "n_heads_distribution": head_counts,
        "mlp_size_distribution": layer_stats["mlp_sizes"],
        "n_kv_group_distribution": layer_stats["kv_groups"],
        "head_per_kv_ratio_distribution": layer_stats["head_group_ratios"],
        "use_concat_heads": globals_cfg.get("use_concat_heads"),
        "layer_mask": globals_cfg.get("layer_mask"),
        "layer_param_distribution": layer_param_estimates,
        "param_center_of_mass": param_com,
        "param_frontness": param_frontness,
        "n_heads_frontness": layer_stats["head_frontness"],
        "mlp_frontness": layer_stats["mlp_frontness"],
        "gqa_frontness": layer_stats["gqa_frontness"],
        "gqa_fraction": layer_stats["gqa_fraction"],
        "identity_fraction": layer_stats["identity_fraction"],
        "identity_frontness": layer_stats["identity_frontness"],
        "qk_v_ratio_distribution": layer_stats["qk_v_ratio_distribution"],
        "qk_v_frontness": layer_stats["qk_v_frontness"],
    }


def analyze_population(population: Population) -> pd.DataFrame:
    """Return a DataFrame with per-individual head statistics for a population."""

    records = []
    evaluations = population.evaluations if getattr(population, "evaluations", None) else []

    for idx, individual in enumerate(getattr(population, "individuals", [])):
        metrics = analyze_individual(individual)
        if not metrics:
            continue
        metrics.update({
            "individual_index": idx,
            "generation": getattr(population, "gen", None),
        })
        if idx < len(evaluations) and evaluations[idx] is not None:
            eval_res = evaluations[idx]
            if hasattr(eval_res, "objs") and len(eval_res.objs) >= 3:
                metrics.update({
                    "val_loss": float(eval_res.objs[0]),
                    "energy_per_token": float(eval_res.objs[1]),
                    "ttft": float(eval_res.objs[2]),
                })
            if hasattr(eval_res, "aux") and isinstance(eval_res.aux, dict):
                metrics.update({
                    "params": eval_res.aux.get("params"),
                    "mem_bytes": eval_res.aux.get("mem_bytes"),
                })
        records.append(metrics)

    return pd.DataFrame.from_records(records)

def plot_scatter(
    file_name_base: str,
    save_path: str,
    start_gen: int,
    end_gen: int,
    x_axis: str = 'params',
    y_axis: str = 'validation_loss',
    color_axis: str = 'generation',
    cmap: str = 'Blues',           # blue (light to dark) sequential colormap
    fix_color_scale: bool = True,  # normalize color scale using global min/max
    point_size: float = 2,       # size of scatter markers (points^2)
    cmap_min: float = 0.2,        # lower bound of colormap (0..1); raise to avoid too-light colors
    cmap_max: float = 1.0         # upper bound of colormap (0..1)
):
    pop_data = []
    generations = list(range(start_gen, end_gen + 1))
    for gen in generations:
        json_file_name = f"{file_name_base}{gen}.json"
        if not os.path.exists(json_file_name):

            print(f"❌ Checkpoint file not found: {json_file_name}")
            exit(f"❌ Checkpoint file not found: {json_file_name}\nPlease ensure all generation checkpoint files are present.")
                
        population = Population.load_checkpoint(json_file_name, from_pkl=False)

        evaluations = [ev for ev in (population.evaluations or []) if ev is not None]
        offspring_evals = [ev for ev in (population.offspring_evaluations or []) if ev is not None]
        all_evals = evaluations + offspring_evals

        individuals = list(population.individuals or []) + list(getattr(population, "offspring", []) or [])

        val_loss_vals = [ev.objs[0] for ev in all_evals if ev.objs and len(ev.objs) > 0]
        energy_vals = [ev.objs[1] for ev in all_evals if ev.objs and len(ev.objs) > 1]
        ttft_vals = [ev.objs[2] for ev in all_evals if ev.objs and len(ev.objs) > 2]

        perplexity = [np.exp(va) for va in val_loss_vals]

        # Ensure all lists have the same length
        min_len = min(len(val_loss_vals), len(energy_vals), len(ttft_vals), len(all_evals), len(individuals))
        
        for i in range(min_len):
            head_metrics = analyze_individual(individuals[i]) if i < len(individuals) else {}
            aux = getattr(all_evals[i], "aux", {}) or {}
            pop_data.append({
                'generation': gen,
                'validation_loss': val_loss_vals[i],
                'energy_per_token': energy_vals[i],
                'ttft': ttft_vals[i],
                'perplexity': perplexity[i],
                'individual_id': i,
                'params': float(aux.get('params', np.nan)) / 1e6 if aux.get('params', np.nan) > 1000 else aux.get('params', np.nan),  # in millions
                'n_heads_mean': head_metrics.get('n_heads_mean'),
                'n_heads_std': head_metrics.get('n_heads_std'),
                'n_heads_min': head_metrics.get('n_heads_min'),
                'n_heads_max': head_metrics.get('n_heads_max'),
                'n_heads_median': head_metrics.get('n_heads_median'),
                'n_heads_distribution': head_metrics.get('n_heads_distribution'),
                'mlp_size_distribution': head_metrics.get('mlp_size_distribution'),
                'n_kv_group_distribution': head_metrics.get('n_kv_group_distribution'),
                'head_per_kv_ratio_distribution': head_metrics.get('head_per_kv_ratio_distribution'),
                'param_frontness': head_metrics.get('param_frontness'),
                'n_heads_frontness': head_metrics.get('n_heads_frontness'),
                'mlp_frontness': head_metrics.get('mlp_frontness'),
                'gqa_frontness': head_metrics.get('gqa_frontness'),
                'gqa_fraction': head_metrics.get('gqa_fraction'),
                'identity_fraction': head_metrics.get('identity_fraction'),
                'identity_frontness': head_metrics.get('identity_frontness'),
                'qk_v_ratio_distribution': head_metrics.get('qk_v_ratio_distribution'),
                'qk_v_frontness': head_metrics.get('qk_v_frontness'),
            })

    df_pop = pd.DataFrame(pop_data)
    design_count = len(df_pop)
    print(f"📊 Total designs plotted: {design_count}")

    # based on the x_axis and y_axis, slect the design on the first pareto front and highlight them
    if x_axis not in df_pop.columns or y_axis not in df_pop.columns or color_axis not in df_pop.columns:
        exit(f"❌ Invalid axis names. Available columns: {df_pop.columns.tolist()}")

    # Select the designs on the first Pareto front
    pareto_front = get_pareto_front(df_pop[x_axis], df_pop[y_axis])
    pareto_df = pd.DataFrame(pareto_front, columns=[x_axis, y_axis])

    print("Detected Pareto front designs:")
    print(pareto_df)

    # Create scatter plot
    fig, ax = plt.subplots(dpi=400)
    # Normalize color scale across all points; enforce symmetric bounds when requested
    if fix_color_scale:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = df_pop[color_axis].min()
        vmax = df_pop[color_axis].max()
    color_values = df_pop[color_axis]
    # Build a truncated colormap to avoid overly light tones
    base_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_vals = base_cmap(np.linspace(max(0.0, cmap_min), min(1.0, cmap_max), 256))
    cmap_vals = _limit_colormap_lightness(cmap_vals)
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{getattr(base_cmap, 'name', 'cmap')}_trunc",
        cmap_vals
    )

    # plot all designs
    scatter = ax.scatter(
        df_pop[x_axis],
        df_pop[y_axis],
        c=color_values,
        cmap=truncated_cmap,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        label='NSGA-II searched architectures'
    )

    # show legend at top-right
    ax.legend(loc='upper right')

    # set axis ranges
    # ax.set_xlim(right=130)
    ax.set_ylim(top=3.3, bottom=2.7)

    # ax.set_xlabel(x_axis)
    ax.set_xlabel("Size (M)")
    ax.set_ylabel("Validation Loss")

    # use blue (light to dark) for color axis (colorbar from full population scatter)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_axis)
    plt.title("Optimizing Accuracy and Size")
    plt.savefig(save_path)
    plt.close()


def main():
    """Main function to create the interactive plots"""
    import os
    
    # take arguments from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Interactive Generational Scatter Plots")
    parser.add_argument("--ckpt_base", type=str, default="ckpts/infi_medium_random/ckpt_offspring_gen", help="Path to the evolution log file")
    parser.add_argument("--start_gen", type=int, default=1, help="Starting generation index (default: 1)")
    # parser.add_argument("--ckpt_gen", type=int, default=50, help="Checkpoint generation index (default: 50)")
    parser.add_argument("--end_gen", type=int, default=91, help="Ending generation index ")
    parser.add_argument("--output", type=str, default="plots/gen_scatter_analysis.png", help="Output png file path")
    parser.add_argument(
        "--frontness",
        type=str,
        default="mlp",
        choices=["params", "n_heads", "mlp", "gqa", "identity", "qk_v_ratio"],
        help="Which frontness metric to use for color coding",
    )
    args = parser.parse_args()
    
    file_name_base = args.ckpt_base
    start_gen = args.start_gen
    end_gen = args.end_gen
    output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frontness_column = {
        "params": "param_frontness",
        "n_heads": "n_heads_frontness",
        "mlp": "mlp_frontness",
        "gqa": "gqa_frontness",
        "identity": "identity_frontness",
        "qk_v_ratio": "qk_v_frontness",
    }[args.frontness]

    plot_scatter(
        file_name_base,
        output_path,
        start_gen,
        end_gen,
        color_axis=frontness_column,
        cmap='RdBu_r',
        cmap_min=0.0,
        cmap_max=1.0,
        fix_color_scale=True,
    )

    # Print completion message
    print(f"✅ Generational scatter plot saved to {output_path}")


if __name__ == "__main__":
    main()