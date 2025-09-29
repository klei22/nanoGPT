"""Utility to explore checkpoint parameters that match a regular expression."""

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from rich import box
from rich.console import Console
from rich.table import Table

# Ensure the repository root is on the import path regardless of CWD
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPTConfig, GPT


ANGLE_METRIC_KEYS: List[Tuple[str, str]] = [
    ("angle_min", "Min angle (deg)"),
    ("angle_mean", "Mean angle (deg)"),
    ("angle_q05", "5% angle (deg)"),
    ("angle_q25", "Q1 angle (deg)"),
    ("angle_median", "Median angle (deg)"),
    ("angle_q75", "Q3 angle (deg)"),
    ("angle_q95", "95% angle (deg)"),
]

COSINE_METRIC_KEYS: List[Tuple[str, str]] = [
    ("cos_min", "Min cosine sim"),
    ("cos_mean", "Mean cosine sim"),
    ("cos_q05", "5% cosine sim"),
    ("cos_q25", "Q1 cosine sim"),
    ("cos_median", "Median cosine sim"),
    ("cos_q75", "Q3 cosine sim"),
    ("cos_q95", "95% cosine sim"),
]

METRIC_DISPLAY: Dict[str, str] = {
    key: label for key, label in [*ANGLE_METRIC_KEYS, *COSINE_METRIC_KEYS]
}

METRIC_CATEGORY: Dict[str, str] = {
    key: "angle" if key.startswith("angle_") else "cosine"
    for key in METRIC_DISPLAY
}

@dataclass
class L2NormStats:
    """Summary statistics for directional L2 norms of a tensor."""

    parameter: str
    axis: int
    axis_size: int
    tensor_shape: Tuple[int, ...]
    num_vectors: int
    min: float
    max: float
    mean: float
    std: float
    kurtosis: float
    histogram_path: Optional[Path]


@dataclass
class DirectionalMetricStats:
    """Summary statistics for pairwise directional angle/cosine metrics."""

    parameter: str
    axis: int
    axis_size: int
    tensor_shape: Tuple[int, ...]
    metric: str
    metric_label: str
    category: str
    num_vectors: int
    min: float
    max: float
    mean: float
    std: float
    q05: float
    q25: float
    median: float
    q75: float
    q95: float
    histogram_path: Optional[Path]


@dataclass
class ColumnSpec:
    """Configuration for a numeric column rendered with heatmap colorization."""

    header: str
    extractor: Callable[[Any], Optional[float]]
    formatter: Callable[[Any], str]
    reverse: bool = False


def _compute_column_ranges(
    data_rows: Iterable[Any], specs: List[ColumnSpec]
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for spec in specs:
        values: List[float] = []
        for row in data_rows:
            value = spec.extractor(row)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                continue
            values.append(numeric_value)
        if values:
            ranges[spec.header] = (min(values), max(values))
        else:
            ranges[spec.header] = (None, None)
    return ranges


def _heatmap_color(
    value: float,
    min_value: Optional[float],
    max_value: Optional[float],
    reverse: bool,
) -> Optional[str]:
    if min_value is None or max_value is None:
        return None
    if math.isclose(max_value, min_value):
        return None

    clamped = max(min(value, max_value), min_value)
    norm = (clamped - min_value) / (max_value - min_value)
    norm = max(0.0, min(1.0, norm))
    if reverse:
        norm = 1.0 - norm

    green = (46, 204, 64)
    red = (255, 65, 54)
    r = int(round(green[0] + (red[0] - green[0]) * norm))
    g = int(round(green[1] + (red[1] - green[1]) * norm))
    b = int(round(green[2] + (red[2] - green[2]) * norm))
    return f"#{r:02x}{g:02x}{b:02x}"


def _format_with_color(
    spec: ColumnSpec,
    data: Any,
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    *,
    colorize: bool,
) -> str:
    text = spec.formatter(data)
    if not colorize:
        return text

    value = spec.extractor(data)
    if value is None:
        return text

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return text

    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return text

    min_value, max_value = ranges.get(spec.header, (None, None))
    color = _heatmap_color(numeric_value, min_value, max_value, spec.reverse)
    if color is None:
        return text
    return f"[{color}]{text}[/]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display statistics for checkpoint parameters matching a regex pattern."
    )
    parser.add_argument("ckpt_path", help="Path to the checkpoint file")
    parser.add_argument("pattern", help="Regular expression used to filter parameter names")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device used to load the checkpoint"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of matching parameters displayed",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension used to compute directional L2 norm statistics. Defaults to the checkpoint configuration.",
    )
    parser.add_argument(
        "--max-l2-rows",
        type=int,
        default=None,
        help="Optional limit on the number of directional L2 norm rows displayed",
    )
    parser.add_argument(
        "--histogram-dir",
        type=Path,
        default=None,
        help="If set, save histograms of L2 norms for matching dimensions to this directory",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=50,
        help="Number of bins used when plotting L2 norm histograms",
    )
    parser.add_argument(
        "--no-colorize",
        dest="colorize",
        action="store_false",
        help="Disable heatmap colorization in stdout tables",
    )
    parser.add_argument(
        "--max-directional-rows",
        type=int,
        default=None,
        help="Optional limit on the number of directional similarity rows displayed",
    )
    parser.add_argument(
        "--directional-chunk-size",
        type=int,
        default=512,
        help="Chunk size used when computing pairwise directional metrics",
    )
    parser.add_argument(
        "--max-directional-vectors",
        type=int,
        default=None,
        help="If set, skip directional metrics when the number of vectors exceeds this threshold",
    )
    parser.set_defaults(colorize=True)
    return parser.parse_args()


def load_state_dict(
    ckpt_path: str, device: str
) -> Tuple[Dict[str, torch.Tensor], GPTConfig]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint.get("model_args")
    state_dict = checkpoint.get("model")

    if model_args is None:
        raise SystemExit("Model arguments not found in checkpoint.")
    if state_dict is None:
        raise SystemExit("Model state dictionary not found in checkpoint.")

    # Instantiate the model so that buffers are materialized in a consistent format
    model = GPT(GPTConfig(**model_args))
    sanitized_state = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            sanitized_state[key[len("_orig_mod."):]] = value
        else:
            sanitized_state[key] = value

    missing, unexpected = model.load_state_dict(sanitized_state, strict=False)
    if missing:
        Console().print(
            f"[yellow]Warning:[/] Missing keys when loading checkpoint: {', '.join(missing)}"
        )
    if unexpected:
        Console().print(
            f"[yellow]Warning:[/] Unexpected keys when loading checkpoint: {', '.join(unexpected)}"
        )

    model.to(device)
    model.eval()

    state_dict = {
        name: parameter.detach().to("cpu") for name, parameter in model.state_dict().items()
    }
    return state_dict, model.config


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        tensor = tensor.detach()
        if tensor.numel() == 0:
            return {
                "shape": tuple(tensor.shape),
                "numel": 0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "kurtosis": float("nan"),
                "median": float("nan"),
                "q1": float("nan"),
                "q3": float("nan"),
                "abs_mean": float("nan"),
                "zeros": 0,
                "zero_pct": float("nan"),
            }

        flat = tensor.to(torch.float32).view(-1)
        numel = flat.numel()
        min_val = flat.min().item()
        max_val = flat.max().item()
        mean = flat.mean().item()
        std = flat.std(unbiased=False).item()
        var = std * std
        if var > 0:
            kurtosis = torch.mean((flat - mean) ** 4).item() / (var ** 2)
        else:
            kurtosis = float("nan")
        median = flat.median().item()
        q1 = torch.quantile(flat, 0.25).item()
        q3 = torch.quantile(flat, 0.75).item()
        abs_mean = flat.abs().mean().item()
        zeros = (flat == 0).sum().item()
        zero_pct = (zeros / numel) * 100 if numel else float("nan")

        return {
            "shape": tuple(tensor.shape),
            "numel": numel,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "std": std,
            "kurtosis": kurtosis,
            "median": median,
            "q1": q1,
            "q3": q3,
            "abs_mean": abs_mean,
            "zeros": zeros,
            "zero_pct": zero_pct,
        }


def update_global_summary(summary: Dict[str, float], stats: Dict[str, float], tensor: torch.Tensor) -> None:
    numel = stats["numel"]
    if numel == 0:
        return

    flat = tensor.to(torch.float32).view(-1)
    summary["numel"] += numel
    summary["sum"] += flat.sum().item()
    summary["sum_sq"] += torch.sum(flat * flat).item()
    summary["abs_sum"] += flat.abs().sum().item()
    summary["zeros"] += stats["zeros"]
    summary["min"] = min(summary["min"], stats["min"])
    summary["max"] = max(summary["max"], stats["max"])


def extract_embedding_vectors(
    tensor: torch.Tensor,
    embedding_dim: Optional[int],
) -> List[Tuple[int, int, Tuple[int, ...], torch.Tensor]]:
    if not embedding_dim:
        return []

    tensor = tensor.detach().to(torch.float32)
    tensor_shape = tuple(tensor.shape)
    results: List[Tuple[int, int, Tuple[int, ...], torch.Tensor]] = []

    for axis, axis_size in enumerate(tensor.shape):
        if axis_size != embedding_dim:
            continue

        moved = tensor.movedim(axis, -1)
        vectors = moved.reshape(-1, embedding_dim)
        if vectors.numel() == 0:
            continue

        results.append((axis, axis_size, tensor_shape, vectors))

    return results


def compute_l2_norm_stats(
    name: str,
    tensor: torch.Tensor,
    embedding_dim: Optional[int],
    histogram_dir: Optional[Path],
    histogram_bins: int,
    *,
    vector_data: Optional[List[Tuple[int, int, Tuple[int, ...], torch.Tensor]]] = None,
) -> List[L2NormStats]:
    if not embedding_dim:
        return []
    results: List[L2NormStats] = []

    vector_entries = vector_data
    if vector_entries is None:
        vector_entries = extract_embedding_vectors(tensor, embedding_dim)

    for axis, axis_size, tensor_shape, vectors in vector_entries:
        norms = torch.linalg.norm(vectors, dim=-1)
        if norms.numel() == 0:
            continue

        min_norm = norms.min().item()
        max_norm = norms.max().item()
        mean_norm = norms.mean().item()
        std_norm = norms.std(unbiased=False).item()
        var_norm = norms.var(unbiased=False).item()
        if var_norm > 0:
            kurtosis = torch.mean((norms - mean_norm) ** 4).item() / (var_norm ** 2)
        else:
            kurtosis = float("nan")

        histogram_path: Optional[Path] = None
        if histogram_dir is not None:
            histogram_path = save_l2_histogram(
                norms,
                histogram_dir,
                name,
                axis,
                tensor_shape=tensor_shape,
                axis_size=axis_size,
                bins=histogram_bins,
            )

        results.append(
            L2NormStats(
                parameter=name,
                axis=axis,
                axis_size=axis_size,
                tensor_shape=tensor_shape,
                num_vectors=norms.numel(),
                min=min_norm,
                max=max_norm,
                mean=mean_norm,
                std=std_norm,
                kurtosis=kurtosis,
                histogram_path=histogram_path,
            )
        )

    return results


def save_l2_histogram(
    norms: torch.Tensor,
    histogram_dir: Path,
    name: str,
    axis: int,
    *,
    tensor_shape: Tuple[int, ...],
    axis_size: int,
    bins: int,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for --histogram-dir but is not installed"
        ) from exc

    histogram_dir.mkdir(parents=True, exist_ok=True)
    shape_token = "x".join(str(dim) for dim in tensor_shape) or "scalar"
    sanitized_name = name.replace("/", "_").replace(".", "_")
    file_name = f"{sanitized_name}_shape{shape_token}_axis{axis}_dim{axis_size}.png"
    file_path = histogram_dir / file_name

    plt.figure(figsize=(8, 4))
    plt.hist(norms.cpu().numpy(), bins=bins, color="#4C72B0", edgecolor="black", alpha=0.8)
    plt.title(
        "\n".join(
            [
                f"L2 norm histogram for {name}",
                f"shape={tensor_shape}, axis={axis}, vector dim={axis_size}",
            ]
        )
    )
    plt.xlabel("L2 norm")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path


def save_metric_histogram(
    values: torch.Tensor,
    histogram_dir: Path,
    name: str,
    axis: int,
    metric_key: str,
    *,
    tensor_shape: Tuple[int, ...],
    axis_size: int,
    bins: int,
    xlabel: str,
    title_prefix: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for --histogram-dir but is not installed"
        ) from exc

    histogram_dir.mkdir(parents=True, exist_ok=True)
    shape_token = "x".join(str(dim) for dim in tensor_shape) or "scalar"
    sanitized_name = name.replace("/", "_").replace(".", "_")
    sanitized_metric = metric_key.replace("/", "_")
    file_name = (
        f"{sanitized_name}_shape{shape_token}_axis{axis}_dim{axis_size}_metric_{sanitized_metric}.png"
    )
    file_path = histogram_dir / file_name

    plt.figure(figsize=(8, 4))
    plt.hist(values.cpu().numpy(), bins=bins, color="#55A868", edgecolor="black", alpha=0.8)
    plt.title(
        "\n".join(
            [
                f"{title_prefix} for {name}",
                f"shape={tensor_shape}, axis={axis}, vector dim={axis_size}",
            ]
        )
    )
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path


def _quantiles(values: torch.Tensor) -> Tuple[float, float, float, float, float]:
    quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=values.dtype, device=values.device)
    q05, q25, median, q75, q95 = torch.quantile(values, quantiles).tolist()
    return q05, q25, median, q75, q95


def compute_directional_metric_stats(
    *,
    name: str,
    axis: int,
    axis_size: int,
    tensor_shape: Tuple[int, ...],
    vectors: torch.Tensor,
    histogram_dir: Optional[Path],
    histogram_bins: int,
    console: Optional[Console],
    chunk_size: int,
    max_vectors: Optional[int],
) -> List[DirectionalMetricStats]:
    num_vectors = vectors.shape[0]
    if num_vectors <= 1:
        return []

    if max_vectors is not None and num_vectors > max_vectors:
        if console is not None:
            console.print(
                f"[yellow]Skipping directional metrics for {name} axis {axis} because there are {num_vectors:,} vectors (limit {max_vectors:,}).[/]"
            )
        return []

    chunk_size = max(1, min(chunk_size, num_vectors))
    vectors = vectors.to(torch.float32)
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    safe_norms = torch.clamp(norms, min=1e-12)
    normalized = vectors / safe_norms

    per_vector_metrics: Dict[str, List[float]] = defaultdict(list)
    for row_start in range(0, num_vectors, chunk_size):
        row_end = min(row_start + chunk_size, num_vectors)
        row_chunk = normalized[row_start:row_end]
        row_size = row_chunk.shape[0]
        row_buffers: List[List[torch.Tensor]] = [[] for _ in range(row_size)]

        row_indices = torch.arange(row_start, row_end)
        for col_start in range(0, num_vectors, chunk_size):
            col_end = min(col_start + chunk_size, num_vectors)
            col_chunk = normalized[col_start:col_end]

            sims = torch.matmul(row_chunk, col_chunk.transpose(0, 1))

            if row_start <= col_end and col_start <= row_end:
                col_indices = torch.arange(col_start, col_end)
                overlap_mask = row_indices.unsqueeze(1) == col_indices.unsqueeze(0)
                sims = sims.masked_fill(overlap_mask, float("nan"))

            for idx in range(row_size):
                row_buffers[idx].append(sims[idx].clone())

        for idx in range(row_size):
            cos_values = torch.cat(row_buffers[idx], dim=0)
            cos_values = cos_values[~torch.isnan(cos_values)]
            if cos_values.numel() == 0:
                continue

            cos_values = torch.clamp(cos_values, -1.0, 1.0)
            angle_values = torch.rad2deg(torch.acos(cos_values))

            angle_min = angle_values.min().item()
            angle_mean = angle_values.mean().item()
            angle_q05, angle_q25, angle_median, angle_q75, angle_q95 = _quantiles(angle_values)

            cos_min = cos_values.min().item()
            cos_mean = cos_values.mean().item()
            cos_q05, cos_q25, cos_median, cos_q75, cos_q95 = _quantiles(cos_values)

            per_vector_metrics["angle_min"].append(angle_min)
            per_vector_metrics["angle_mean"].append(angle_mean)
            per_vector_metrics["angle_q05"].append(angle_q05)
            per_vector_metrics["angle_q25"].append(angle_q25)
            per_vector_metrics["angle_median"].append(angle_median)
            per_vector_metrics["angle_q75"].append(angle_q75)
            per_vector_metrics["angle_q95"].append(angle_q95)

            per_vector_metrics["cos_min"].append(cos_min)
            per_vector_metrics["cos_mean"].append(cos_mean)
            per_vector_metrics["cos_q05"].append(cos_q05)
            per_vector_metrics["cos_q25"].append(cos_q25)
            per_vector_metrics["cos_median"].append(cos_median)
            per_vector_metrics["cos_q75"].append(cos_q75)
            per_vector_metrics["cos_q95"].append(cos_q95)

    results: List[DirectionalMetricStats] = []
    for metric_key, values in per_vector_metrics.items():
        metric_tensor = torch.tensor(values, dtype=torch.float32)
        if metric_tensor.numel() == 0:
            continue

        min_val = metric_tensor.min().item()
        max_val = metric_tensor.max().item()
        mean_val = metric_tensor.mean().item()
        std_val = metric_tensor.std(unbiased=False).item() if metric_tensor.numel() > 1 else 0.0
        q05_val, q25_val, median_val, q75_val, q95_val = _quantiles(metric_tensor)

        histogram_path: Optional[Path] = None
        if histogram_dir is not None:
            category = METRIC_CATEGORY.get(metric_key, "unknown")
            if category == "angle":
                title_prefix = "Directional angle statistic histogram"
            elif category == "cosine":
                title_prefix = "Directional cosine statistic histogram"
            else:
                title_prefix = "Directional metric histogram"
            histogram_path = save_metric_histogram(
                metric_tensor,
                histogram_dir,
                name,
                axis,
                metric_key,
                tensor_shape=tensor_shape,
                axis_size=axis_size,
                bins=histogram_bins,
                xlabel=METRIC_DISPLAY.get(metric_key, metric_key),
                title_prefix=title_prefix,
            )

        results.append(
            DirectionalMetricStats(
                parameter=name,
                axis=axis,
                axis_size=axis_size,
                tensor_shape=tensor_shape,
                metric=metric_key,
                metric_label=METRIC_DISPLAY.get(metric_key, metric_key),
                category=METRIC_CATEGORY.get(metric_key, "unknown"),
                num_vectors=len(values),
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
                q05=q05_val,
                q25=q25_val,
                median=median_val,
                q75=q75_val,
                q95=q95_val,
                histogram_path=histogram_path,
            )
        )

    return sorted(
        results,
        key=lambda item: (
            item.parameter,
            item.axis,
            0 if item.metric in METRIC_DISPLAY else 1,
            item.metric,
        ),
    )


def register_vectors_for_groups(
    name: str,
    vectors: torch.Tensor,
    registry: Dict[str, List[torch.Tensor]],
) -> None:
    is_wte = name.endswith("wte.weight")
    is_attn_c_proj = ".attn.c_proj.weight" in name
    is_mlp_c_proj = ".mlp.c_proj.weight" in name

    registry["all_vectors"].append(vectors)

    if is_wte:
        registry["wte"].append(vectors)
        registry["cproj_with_wte"].append(vectors)

    if is_attn_c_proj:
        registry["attn_c_proj"].append(vectors)
        registry["all_c_proj"].append(vectors)
        registry["cproj_with_wte"].append(vectors)

    if is_mlp_c_proj:
        registry["mlp_c_proj"].append(vectors)
        registry["all_c_proj"].append(vectors)
        registry["cproj_with_wte"].append(vectors)


def render_table(
    rows: Iterable[Tuple[str, Dict[str, float]]],
    max_rows: int | None,
    *,
    colorize: bool,
) -> None:
    rows = list(rows)
    total_rows = len(rows)
    display_rows = rows if max_rows is None else rows[:max_rows]
    stats_rows = [stats for _, stats in rows]

    column_specs: List[ColumnSpec] = [
        ColumnSpec(
            "# Elem",
            extractor=lambda stats: float(stats["numel"]),
            formatter=lambda stats: f"{int(stats['numel']):,}",
        ),
        ColumnSpec(
            "Min",
            extractor=lambda stats: float(stats["min"]),
            formatter=lambda stats: f"{stats['min']:.6g}",
            reverse=True,
        ),
        ColumnSpec(
            "Max",
            extractor=lambda stats: float(stats["max"]),
            formatter=lambda stats: f"{stats['max']:.6g}",
        ),
        ColumnSpec(
            "Mean",
            extractor=lambda stats: float(stats["mean"]),
            formatter=lambda stats: f"{stats['mean']:.6g}",
        ),
        ColumnSpec(
            "Std",
            extractor=lambda stats: float(stats["std"]),
            formatter=lambda stats: f"{stats['std']:.6g}",
        ),
        ColumnSpec(
            "Kurtosis",
            extractor=lambda stats: None
            if math.isnan(stats["kurtosis"])
            else float(stats["kurtosis"]),
            formatter=lambda stats: f"{stats['kurtosis']:.6g}"
            if not math.isnan(stats["kurtosis"])
            else "n/a",
        ),
        ColumnSpec(
            "Median",
            extractor=lambda stats: float(stats["median"]),
            formatter=lambda stats: f"{stats['median']:.6g}",
        ),
        ColumnSpec(
            "Q1",
            extractor=lambda stats: float(stats["q1"]),
            formatter=lambda stats: f"{stats['q1']:.6g}",
            reverse=True,
        ),
        ColumnSpec(
            "Q3",
            extractor=lambda stats: float(stats["q3"]),
            formatter=lambda stats: f"{stats['q3']:.6g}",
        ),
        ColumnSpec(
            "|Mean|",
            extractor=lambda stats: float(stats["abs_mean"]),
            formatter=lambda stats: f"{stats['abs_mean']:.6g}",
        ),
        ColumnSpec(
            "Zeros %",
            extractor=lambda stats: None
            if math.isnan(stats["zero_pct"])
            else float(stats["zero_pct"]),
            formatter=lambda stats: f"{stats['zero_pct']:.3g}%"
            if not math.isnan(stats["zero_pct"])
            else "n/a",
        ),
    ]

    column_ranges = _compute_column_ranges(stats_rows, column_specs)

    table = Table(
        title="Parameter statistics",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("Parameter", overflow="fold")
    table.add_column("Shape", justify="center")
    table.add_column("# Elem", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Kurtosis", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Q1", justify="right")
    table.add_column("Q3", justify="right")
    table.add_column("|Mean|", justify="right")
    table.add_column("Zeros %", justify="right")

    count = 0
    for name, stats in display_rows:
        formatted_cells = [
            _format_with_color(spec, stats, column_ranges, colorize=colorize)
            for spec in column_specs
        ]
        table.add_row(
            name,
            str(stats["shape"]),
            *formatted_cells,
        )
        count += 1

    console = Console()
    console.print(table)
    if max_rows is not None and total_rows > max_rows:
        console.print(
            f"[dim]Displayed {count} of {total_rows} rows (limited by --max-rows). Use a larger value to see more.[/]"
        )


def render_l2_table(
    rows: List[L2NormStats],
    max_rows: int | None,
    *,
    colorize: bool,
) -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda item: (item.parameter, item.axis))
    total_rows = len(rows)
    display_rows = rows if max_rows is None else rows[:max_rows]
    has_histograms = any(row.histogram_path is not None for row in rows)

    column_specs: List[ColumnSpec] = [
        ColumnSpec(
            "# Vectors",
            extractor=lambda row: float(row.num_vectors),
            formatter=lambda row: f"{row.num_vectors:,}",
        ),
        ColumnSpec(
            "Min",
            extractor=lambda row: float(row.min),
            formatter=lambda row: f"{row.min:.6g}",
            reverse=True,
        ),
        ColumnSpec(
            "Max",
            extractor=lambda row: float(row.max),
            formatter=lambda row: f"{row.max:.6g}",
        ),
        ColumnSpec(
            "Mean",
            extractor=lambda row: float(row.mean),
            formatter=lambda row: f"{row.mean:.6g}",
        ),
        ColumnSpec(
            "Std",
            extractor=lambda row: float(row.std),
            formatter=lambda row: f"{row.std:.6g}",
        ),
        ColumnSpec(
            "Kurtosis",
            extractor=lambda row: None if math.isnan(row.kurtosis) else float(row.kurtosis),
            formatter=lambda row: f"{row.kurtosis:.6g}" if not math.isnan(row.kurtosis) else "n/a",
        ),
    ]

    column_ranges = _compute_column_ranges(rows, column_specs)

    table = Table(
        title="Directional L2 norm statistics",
        box=box.SIMPLE_HEAVY,
        header_style="bold blue",
        show_lines=False,
    )
    table.add_column("Parameter", overflow="fold")
    table.add_column("Shape", justify="center")
    table.add_column("Axis", justify="center")
    table.add_column("# Vectors", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Kurtosis", justify="right")
    if has_histograms:
        table.add_column("Histogram", overflow="fold")

    for row in display_rows:
        formatted_cells = [
            _format_with_color(spec, row, column_ranges, colorize=colorize)
            for spec in column_specs
        ]
        values = [
            row.parameter,
            str(row.tensor_shape),
            f"axis {row.axis} (vector dim={row.axis_size})",
            *formatted_cells,
        ]
        if has_histograms:
            values.append(str(row.histogram_path) if row.histogram_path else "—")
        table.add_row(*values)

    console = Console()
    console.print(table)
    if max_rows is not None and total_rows > max_rows:
        console.print(
            f"[dim]Displayed {len(display_rows)} of {total_rows} rows (limited by --max-l2-rows). Use a larger value to see more.[/]"
        )


def render_directional_metric_table(
    rows: List[DirectionalMetricStats],
    max_rows: Optional[int],
    *,
    colorize: bool,
) -> None:
    if not rows:
        return

    rows = sorted(
        rows,
        key=lambda item: (
            item.parameter,
            item.axis,
            0 if item.category == "angle" else 1,
            item.metric,
        ),
    )

    total_rows = len(rows)
    display_rows = rows if max_rows is None else rows[:max_rows]
    has_histograms = any(row.histogram_path is not None for row in rows)

    column_specs: List[ColumnSpec] = [
        ColumnSpec(
            "# Vectors",
            extractor=lambda row: float(row.num_vectors),
            formatter=lambda row: f"{row.num_vectors:,}",
        ),
        ColumnSpec(
            "Min",
            extractor=lambda row: float(row.min),
            formatter=lambda row: f"{row.min:.6g}",
            reverse=True,
        ),
        ColumnSpec(
            "Max",
            extractor=lambda row: float(row.max),
            formatter=lambda row: f"{row.max:.6g}",
        ),
        ColumnSpec(
            "Mean",
            extractor=lambda row: float(row.mean),
            formatter=lambda row: f"{row.mean:.6g}",
        ),
        ColumnSpec(
            "Std",
            extractor=lambda row: float(row.std),
            formatter=lambda row: f"{row.std:.6g}",
        ),
        ColumnSpec(
            "Q05",
            extractor=lambda row: float(row.q05),
            formatter=lambda row: f"{row.q05:.6g}",
            reverse=True,
        ),
        ColumnSpec(
            "Q25",
            extractor=lambda row: float(row.q25),
            formatter=lambda row: f"{row.q25:.6g}",
            reverse=True,
        ),
        ColumnSpec(
            "Median",
            extractor=lambda row: float(row.median),
            formatter=lambda row: f"{row.median:.6g}",
        ),
        ColumnSpec(
            "Q75",
            extractor=lambda row: float(row.q75),
            formatter=lambda row: f"{row.q75:.6g}",
        ),
        ColumnSpec(
            "Q95",
            extractor=lambda row: float(row.q95),
            formatter=lambda row: f"{row.q95:.6g}",
        ),
    ]

    column_ranges = _compute_column_ranges(rows, column_specs)

    table = Table(
        title="Directional similarity statistics",
        box=box.SIMPLE_HEAVY,
        header_style="bold green",
        show_lines=False,
    )
    table.add_column("Parameter", overflow="fold")
    table.add_column("Shape", justify="center")
    table.add_column("Axis", justify="center")
    table.add_column("Metric", justify="left")
    table.add_column("Category", justify="left")
    table.add_column("# Vectors", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Q05", justify="right")
    table.add_column("Q25", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Q75", justify="right")
    table.add_column("Q95", justify="right")
    if has_histograms:
        table.add_column("Histogram", overflow="fold")

    for row in display_rows:
        formatted_cells = [
            _format_with_color(spec, row, column_ranges, colorize=colorize)
            for spec in column_specs
        ]

        values = [
            row.parameter,
            str(row.tensor_shape),
            f"axis {row.axis} (vector dim={row.axis_size})",
            row.metric_label,
            row.category,
            *formatted_cells,
        ]
        if has_histograms:
            values.append(str(row.histogram_path) if row.histogram_path else "—")
        table.add_row(*values)

    console = Console()
    console.print(table)
    if max_rows is not None and total_rows > max_rows:
        console.print(
            f"[dim]Displayed {len(display_rows)} of {total_rows} rows (limited by --max-directional-rows). Use a larger value to see more.[/]"
        )


def render_summary(summary: Dict[str, float], matched: int) -> None:
    table = Table(title="Aggregate statistics", box=box.SQUARE)
    table.add_column("Metric", style="bold magenta")
    table.add_column("Value", style="bold cyan", justify="right")

    table.add_row("Matched parameters", str(matched))
    table.add_row("Total elements", f"{int(summary['numel']):,}")

    if summary["numel"]:
        mean = summary["sum"] / summary["numel"]
        var = summary["sum_sq"] / summary["numel"] - mean * mean
        std = math.sqrt(max(var, 0.0))
        abs_mean = summary["abs_sum"] / summary["numel"]
        zero_pct = (summary["zeros"] / summary["numel"]) * 100
        table.add_row("Min", f"{summary['min']:.6g}")
        table.add_row("Max", f"{summary['max']:.6g}")
        table.add_row("Mean", f"{mean:.6g}")
        table.add_row("Std", f"{std:.6g}")
        table.add_row("|Mean|", f"{abs_mean:.6g}")
        table.add_row("Zeros %", f"{zero_pct:.3g}%")
    else:
        table.add_row("Note", "No elements matched the provided pattern")

    Console().print(table)


def main() -> None:
    args = parse_args()
    console = Console()

    try:
        state_dict, config = load_state_dict(args.ckpt_path, args.device)
    except FileNotFoundError as exc:
        raise SystemExit(f"Checkpoint not found: {exc}") from exc

    pattern = re.compile(args.pattern)

    embedding_dim = args.embedding_dim if args.embedding_dim else getattr(config, "n_embd", None)
    if embedding_dim is None:
        console.print(
            "[yellow]Embedding dimension not found in checkpoint configuration; skipping L2 norm statistics.[/]"
        )

    histogram_dir = args.histogram_dir
    if histogram_dir is not None:
        histogram_dir = histogram_dir.expanduser().resolve()

    rows = []
    l2_rows: List[L2NormStats] = []
    directional_rows: List[DirectionalMetricStats] = []
    summary = {
        "numel": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "abs_sum": 0.0,
        "zeros": 0,
        "min": float("inf"),
        "max": float("-inf"),
    }

    group_registry: Dict[str, List[torch.Tensor]] = defaultdict(list)
    group_display_names = {
        "wte": "wte",
        "attn_c_proj": "attn c_proj (all)",
        "mlp_c_proj": "mlp c_proj (all)",
        "all_c_proj": "all c_proj",
        "cproj_with_wte": "c_proj + wte",
        "all_vectors": "all vectors",
    }

    for name, tensor in state_dict.items():
        if pattern.search(name):
            stats = tensor_stats(tensor)
            rows.append((name, stats))
            update_global_summary(summary, stats, tensor)
            if embedding_dim:
                vector_entries = extract_embedding_vectors(tensor, embedding_dim)
                if vector_entries:
                    l2_rows.extend(
                        compute_l2_norm_stats(
                            name,
                            tensor,
                            embedding_dim=embedding_dim,
                            histogram_dir=histogram_dir,
                            histogram_bins=args.histogram_bins,
                            vector_data=vector_entries,
                        )
                    )
                    for axis, axis_size, tensor_shape, vectors in vector_entries:
                        register_vectors_for_groups(name, vectors, group_registry)
                        directional_rows.extend(
                            compute_directional_metric_stats(
                                name=name,
                                axis=axis,
                                axis_size=axis_size,
                                tensor_shape=tensor_shape,
                                vectors=vectors,
                                histogram_dir=histogram_dir,
                                histogram_bins=args.histogram_bins,
                                console=console,
                                chunk_size=args.directional_chunk_size,
                                max_vectors=args.max_directional_vectors,
                            )
                        )

    if not rows:
        console.print("[red]No parameters matched the provided pattern.[/]")
        return

    rows.sort(key=lambda item: item[0])
    render_table(rows, args.max_rows, colorize=args.colorize)
    render_summary(summary, matched=len(rows))
    if embedding_dim:
        for group_key, display_name in group_display_names.items():
            vectors_list = group_registry.get(group_key)
            if not vectors_list:
                continue
            combined = torch.cat(vectors_list, dim=0).to(torch.float32)
            if combined.numel() == 0:
                continue

            group_name = f"[Group] {display_name}"
            axis_index = combined.dim() - 1
            tensor_shape = tuple(combined.shape)
            vector_data = [(axis_index, embedding_dim, tensor_shape, combined)]

            l2_rows.extend(
                compute_l2_norm_stats(
                    group_name,
                    combined,
                    embedding_dim=embedding_dim,
                    histogram_dir=histogram_dir,
                    histogram_bins=args.histogram_bins,
                    vector_data=vector_data,
                )
            )

            directional_rows.extend(
                compute_directional_metric_stats(
                    name=group_name,
                    axis=axis_index,
                    axis_size=embedding_dim,
                    tensor_shape=tensor_shape,
                    vectors=combined,
                    histogram_dir=histogram_dir,
                    histogram_bins=args.histogram_bins,
                    console=console,
                    chunk_size=args.directional_chunk_size,
                    max_vectors=args.max_directional_vectors,
                )
            )

        if l2_rows:
            render_l2_table(l2_rows, args.max_l2_rows, colorize=args.colorize)

        if directional_rows:
            render_directional_metric_table(
                directional_rows,
                args.max_directional_rows,
                colorize=args.colorize,
            )
    else:
        if l2_rows:
            render_l2_table(l2_rows, args.max_l2_rows, colorize=args.colorize)


if __name__ == "__main__":
    main()

