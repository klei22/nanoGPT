"""Utility to explore checkpoint parameters that match a regular expression."""

import argparse
import csv
import math
import re
import sys
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
class ColumnSpec:
    """Configuration for a numeric column rendered with heatmap colorization."""

    header: str
    extractor: Callable[[Any], Optional[float]]
    formatter: Callable[[Any], str]
    reverse: bool = False


@dataclass
class PairwiseMetricStats:
    """Summary statistics for pairwise angle/cosine metrics."""

    parameter: str
    axis: int
    axis_size: int
    tensor_shape: Tuple[int, ...]
    num_vectors: int
    metric: str
    units: str
    min: float
    max: float
    mean: float
    std: float
    q05: float
    q1: float
    median: float
    q3: float
    q95: float
    histogram_path: Optional[Path]


@dataclass
class AngleDifferenceStats:
    """Summary statistics for per-vector angular differences between checkpoints."""

    parameter: str
    axis: int
    axis_size: int
    tensor_shape: Tuple[int, ...]
    num_vectors: int
    min: float
    max: float
    mean: float
    std: float
    q05: float
    q1: float
    median: float
    q3: float
    q95: float
    histogram_path: Optional[Path]


@dataclass
class CheckpointAnalysis:
    """Container for single-checkpoint analysis outputs."""

    path: Path
    rows: List[Tuple[str, Dict[str, float]]]
    summary: Dict[str, float]
    l2_rows: List[L2NormStats]
    pairwise_rows: List[PairwiseMetricStats]
    group_pairwise_rows: List[PairwiseMetricStats]


PAIRWISE_METRIC_INFO: Dict[str, Tuple[str, str]] = {
    "angle_min": ("Min angle", "angle"),
    "angle_mean": ("Mean angle", "angle"),
    "angle_q05": ("5th pct angle", "angle"),
    "angle_q1": ("25th pct angle", "angle"),
    "angle_median": ("Median angle", "angle"),
    "angle_q3": ("75th pct angle", "angle"),
    "angle_q95": ("95th pct angle", "angle"),
    "cos_min": ("Min cosine similarity", "cosine"),
    "cos_mean": ("Mean cosine similarity", "cosine"),
    "cos_q05": ("5th pct cosine similarity", "cosine"),
    "cos_q1": ("25th pct cosine similarity", "cosine"),
    "cos_median": ("Median cosine similarity", "cosine"),
    "cos_q3": ("75th pct cosine similarity", "cosine"),
    "cos_q95": ("95th pct cosine similarity", "cosine"),
    "cos_max": ("Max cosine similarity", "cosine"),
}


def iter_vector_views(
    tensor: torch.Tensor, embedding_dim: Optional[int]
) -> Iterable[Tuple[int, int, torch.Tensor]]:
    """Yield per-axis views reshaped into (num_vectors, embedding_dim)."""

    if embedding_dim is None:
        return

    tensor = tensor.detach().to(torch.float32)
    for axis, axis_size in enumerate(tensor.shape):
        if axis_size != embedding_dim:
            continue
        moved = tensor.movedim(axis, -1)
        vectors = moved.reshape(-1, embedding_dim)
        if vectors.numel() == 0:
            continue
        yield axis, axis_size, vectors


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
    parser.add_argument(
        "ckpt_paths",
        nargs="+",
        help="Path to the checkpoint file(s). Provide one path for single checkpoint analysis or two paths to compare checkpoints.",
    )
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
        "--max-angle-rows",
        type=int,
        default=None,
        help="Optional limit on the number of per-tensor angle comparison rows displayed when comparing checkpoints.",
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
        "--pairwise-limit",
        type=int,
        default=0,
        help=(
            "Maximum number of vectors per tensor/group when computing pairwise "
            "angle/cosine statistics. Set to 0 to disable the limit."
        ),
    )
    parser.add_argument(
        "--angle-units",
        choices=["degrees", "radians"],
        default="degrees",
        help=(
            "Units used when reporting pairwise angle statistics."
            " Defaults to degrees."
        ),
    )
    parser.add_argument(
        "--max-pairwise-rows",
        type=int,
        default=None,
        help="Optional limit on the number of pairwise metric rows displayed",
    )
    parser.add_argument(
        "--angle-csv",
        type=Path,
        default=Path("angle_differences.csv"),
        help="Path to the CSV file where per-vector angular differences are written when comparing checkpoints.",
    )
    parser.add_argument(
        "--no-colorize",
        dest="colorize",
        action="store_false",
        help="Disable heatmap colorization in stdout tables",
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


def compute_l2_norm_stats_for_vectors(
    name: str,
    tensor_shape: Tuple[int, ...],
    axis: int,
    axis_size: int,
    vectors: torch.Tensor,
    histogram_dir: Optional[Path],
    histogram_bins: int,
) -> Optional[L2NormStats]:
    if vectors.numel() == 0:
        return None

    norms = torch.linalg.norm(vectors, dim=-1)
    if norms.numel() == 0:
        return None

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

    return L2NormStats(
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


def summarize_metric_values(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().to(torch.float64)
    finite_mask = torch.isfinite(values)
    if finite_mask.numel() == 0 or not torch.any(finite_mask):
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "q05": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "q95": float("nan"),
        }

    finite_values = values[finite_mask]
    if finite_values.numel() == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "q05": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "q95": float("nan"),
        }

    min_val = finite_values.min().item()
    max_val = finite_values.max().item()
    mean_val = finite_values.mean().item()
    std_val = finite_values.std(unbiased=False).item()
    q05_val = torch.quantile(finite_values, 0.05).item()
    q1_val = torch.quantile(finite_values, 0.25).item()
    median_val = torch.quantile(finite_values, 0.5).item()
    q3_val = torch.quantile(finite_values, 0.75).item()
    q95_val = torch.quantile(finite_values, 0.95).item()

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "q05": q05_val,
        "q1": q1_val,
        "median": median_val,
        "q3": q3_val,
        "q95": q95_val,
    }


def compute_vector_angle_metrics(
    vectors_a: torch.Tensor,
    vectors_b: torch.Tensor,
    *,
    angle_units: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors_a = vectors_a.detach().to(torch.float64)
    vectors_b = vectors_b.detach().to(torch.float64)
    norms_a = torch.linalg.norm(vectors_a, dim=-1)
    norms_b = torch.linalg.norm(vectors_b, dim=-1)
    denom = norms_a * norms_b
    dot = torch.sum(vectors_a * vectors_b, dim=-1)

    cosine = torch.full_like(dot, float("nan"))
    valid_mask = denom > 0
    if valid_mask.any():
        cos_values = (dot[valid_mask] / denom[valid_mask]).clamp(-1.0, 1.0)
        cosine[valid_mask] = cos_values

    angles = torch.full_like(cosine, float("nan"))
    if valid_mask.any():
        valid_angles = torch.acos(cosine[valid_mask].clamp(-1.0, 1.0))
        if angle_units == "degrees":
            valid_angles = torch.rad2deg(valid_angles)
        elif angle_units != "radians":
            raise ValueError(f"Unsupported angle unit: {angle_units}")
        angles[valid_mask] = valid_angles

    return angles, cosine, norms_a, norms_b, valid_mask


def save_pairwise_histogram(
    values: torch.Tensor,
    histogram_dir: Path,
    name: str,
    axis: int,
    *,
    tensor_shape: Tuple[int, ...],
    axis_size: int,
    metric: str,
    units: str,
    bins: int,
    histogram_prefix: Optional[str] = None,
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
    axis_token = str(axis) if axis >= 0 else "group"
    base_name = histogram_prefix if histogram_prefix is not None else name
    sanitized_name = base_name.replace("/", "_").replace(".", "_")
    metric_token = metric.replace(" ", "_").replace("/", "_")
    file_name = (
        f"{sanitized_name}_shape{shape_token}_axis{axis_token}_dim{axis_size}_{metric_token}.png"
    )
    file_path = histogram_dir / file_name

    plt.figure(figsize=(8, 4))
    plt.hist(values.cpu().numpy(), bins=bins, color="#55A868", edgecolor="black", alpha=0.8)
    plt.title(
        "\n".join(
            [
                f"Pairwise {metric} histogram for {base_name}",
                f"shape={tensor_shape}, axis={axis_token}, vector dim={axis_size}",
            ]
        )
    )
    if units == "cosine":
        xlabel = "Cosine similarity"
    elif units == "degrees":
        xlabel = "Angle (degrees)"
    else:
        xlabel = "Angle (radians)"
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path


def compute_pairwise_metrics(
    name: str,
    tensor_shape: Tuple[int, ...],
    axis: int,
    axis_size: int,
    vectors: torch.Tensor,
    histogram_dir: Optional[Path],
    histogram_bins: int,
    *,
    histogram_prefix: Optional[str] = None,
    angle_units: str,
) -> List[PairwiseMetricStats]:
    num_vectors = vectors.shape[0]
    if num_vectors <= 1:
        return []

    normalized = torch.nn.functional.normalize(vectors, dim=-1)
    # Zero vectors lead to NaNs; replace them with zeros to keep cosine finite.
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    cosine_matrix = torch.matmul(normalized, normalized.T)
    cosine_matrix = cosine_matrix.clamp(-1.0, 1.0)

    mask = ~torch.eye(num_vectors, dtype=torch.bool, device=cosine_matrix.device)
    if mask.sum() == 0:
        return []

    cosine_values = cosine_matrix.masked_select(mask).view(num_vectors, num_vectors - 1)

    cos_min = cosine_values.min(dim=1).values
    cos_max = cosine_values.max(dim=1).values
    cos_mean = cosine_values.mean(dim=1)
    cos_q05 = torch.quantile(cosine_values, 0.05, dim=1)
    cos_q1 = torch.quantile(cosine_values, 0.25, dim=1)
    cos_median = torch.quantile(cosine_values, 0.5, dim=1)
    cos_q3 = torch.quantile(cosine_values, 0.75, dim=1)
    cos_q95 = torch.quantile(cosine_values, 0.95, dim=1)

    angle_values = torch.acos(cosine_values.clamp(-1.0, 1.0))
    if angle_units == "degrees":
        angle_values = torch.rad2deg(angle_values)
    elif angle_units != "radians":
        raise ValueError(f"Unsupported angle unit: {angle_units}")
    angle_min = angle_values.min(dim=1).values
    angle_mean = angle_values.mean(dim=1)
    angle_q05 = torch.quantile(angle_values, 0.05, dim=1)
    angle_q1 = torch.quantile(angle_values, 0.25, dim=1)
    angle_median = torch.quantile(angle_values, 0.5, dim=1)
    angle_q3 = torch.quantile(angle_values, 0.75, dim=1)
    angle_q95 = torch.quantile(angle_values, 0.95, dim=1)

    metric_values: Dict[str, torch.Tensor] = {
        "angle_min": angle_min,
        "angle_mean": angle_mean,
        "angle_q05": angle_q05,
        "angle_q1": angle_q1,
        "angle_median": angle_median,
        "angle_q3": angle_q3,
        "angle_q95": angle_q95,
        "cos_min": cos_min,
        "cos_mean": cos_mean,
        "cos_q05": cos_q05,
        "cos_q1": cos_q1,
        "cos_median": cos_median,
        "cos_q3": cos_q3,
        "cos_q95": cos_q95,
        "cos_max": cos_max,
    }

    results: List[PairwiseMetricStats] = []
    for metric, tensor_values in metric_values.items():
        label, unit_kind = PAIRWISE_METRIC_INFO.get(metric, (metric, ""))
        units = angle_units if unit_kind == "angle" else unit_kind
        summary = summarize_metric_values(tensor_values)
        histogram_path: Optional[Path] = None
        if histogram_dir is not None:
            histogram_path = save_pairwise_histogram(
                tensor_values,
                histogram_dir,
                name,
                axis,
                tensor_shape=tensor_shape,
                axis_size=axis_size,
                metric=label,
                units=units,
                bins=histogram_bins,
                histogram_prefix=histogram_prefix,
            )

        results.append(
            PairwiseMetricStats(
                parameter=name,
                axis=axis,
                axis_size=axis_size,
                tensor_shape=tensor_shape,
                num_vectors=num_vectors,
                metric=metric,
                units=units,
                min=summary["min"],
                max=summary["max"],
                mean=summary["mean"],
                std=summary["std"],
                q05=summary["q05"],
                q1=summary["q1"],
                median=summary["median"],
                q3=summary["q3"],
                q95=summary["q95"],
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


def save_angle_histogram(
    values: torch.Tensor,
    histogram_dir: Path,
    name: str,
    axis: int,
    *,
    tensor_shape: Tuple[int, ...],
    axis_size: int,
    angle_units: str,
    bins: int,
    histogram_prefix: Optional[str] = None,
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
    axis_token = "overall" if axis < 0 else str(axis)
    base_name = histogram_prefix if histogram_prefix is not None else name
    sanitized_name = base_name.replace("/", "_").replace(".", "_")
    unit_token = "deg" if angle_units == "degrees" else "rad"
    file_name = (
        f"{sanitized_name}_shape{shape_token}_axis{axis_token}_dim{axis_size}_angle_{unit_token}.png"
    )
    file_path = histogram_dir / file_name

    plt.figure(figsize=(8, 4))
    plt.hist(values.cpu().numpy(), bins=bins, color="#C44E52", edgecolor="black", alpha=0.8)
    plt.title(
        "\n".join(
            [
                f"Angular difference histogram for {base_name}",
                f"shape={tensor_shape}, axis={axis_token}, vector dim={axis_size}",
            ]
        )
    )
    xlabel = "Angle (degrees)" if angle_units == "degrees" else "Angle (radians)"
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path


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


def render_pairwise_table(
    rows: List[PairwiseMetricStats],
    max_rows: Optional[int],
    *,
    colorize: bool,
    title: str,
) -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda item: (item.parameter, item.axis, item.metric))
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
        ),
        ColumnSpec(
            "Q1",
            extractor=lambda row: float(row.q1),
            formatter=lambda row: f"{row.q1:.6g}",
        ),
        ColumnSpec(
            "Median",
            extractor=lambda row: float(row.median),
            formatter=lambda row: f"{row.median:.6g}",
        ),
        ColumnSpec(
            "Q3",
            extractor=lambda row: float(row.q3),
            formatter=lambda row: f"{row.q3:.6g}",
        ),
        ColumnSpec(
            "Q95",
            extractor=lambda row: float(row.q95),
            formatter=lambda row: f"{row.q95:.6g}",
        ),
    ]

    column_ranges = _compute_column_ranges(rows, column_specs)

    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        header_style="bold green",
        show_lines=False,
    )
    table.add_column("Parameter", overflow="fold")
    table.add_column("Shape", justify="center")
    table.add_column("Axis", justify="center")
    table.add_column("Metric", justify="left")
    table.add_column("Units", justify="center")
    table.add_column("# Vectors", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Q05", justify="right")
    table.add_column("Q1", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Q3", justify="right")
    table.add_column("Q95", justify="right")
    if has_histograms:
        table.add_column("Histogram", overflow="fold")

    for row in display_rows:
        formatted_cells = [
            _format_with_color(spec, row, column_ranges, colorize=colorize)
            for spec in column_specs
        ]
        metric_label = PAIRWISE_METRIC_INFO.get(row.metric, (row.metric, ""))[0]
        axis_label = (
            f"axis {row.axis} (vector dim={row.axis_size})"
            if row.axis >= 0
            else f"group (vector dim={row.axis_size})"
        )
        values = [
            row.parameter,
            str(row.tensor_shape),
            axis_label,
            metric_label,
            row.units,
            *formatted_cells,
        ]
        if has_histograms:
            values.append(str(row.histogram_path) if row.histogram_path else "—")
        table.add_row(*values)

    console = Console()
    console.print(table)
    if max_rows is not None and total_rows > max_rows:
        console.print(
            f"[dim]Displayed {len(display_rows)} of {total_rows} rows (limited by --max-pairwise-rows). Use a larger value to see more.[/]"
        )


def analyze_checkpoint(
    path: Path,
    state_dict: Dict[str, torch.Tensor],
    pattern: re.Pattern,
    *,
    embedding_dim: Optional[int],
    histogram_dir: Optional[Path],
    histogram_bins: int,
    pairwise_limit: int,
    angle_units: str,
    histogram_prefix_root: Optional[str] = None,
) -> CheckpointAnalysis:
    console = Console()
    rows: List[Tuple[str, Dict[str, float]]] = []
    l2_rows: List[L2NormStats] = []
    pairwise_rows: List[PairwiseMetricStats] = []
    group_pairwise_rows: List[PairwiseMetricStats] = []
    group_vectors: Dict[str, List[torch.Tensor]] = {
        "wte": [],
        "attn_c_proj": [],
        "mlp_c_proj": [],
        "all_vectors": [],
    }

    summary = {
        "numel": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "abs_sum": 0.0,
        "zeros": 0,
        "min": float("inf"),
        "max": float("-inf"),
    }

    for name, tensor in state_dict.items():
        if not pattern.search(name):
            continue

        stats = tensor_stats(tensor)
        rows.append((name, stats))
        update_global_summary(summary, stats, tensor)

        if embedding_dim:
            tensor_shape = tuple(tensor.shape)
            for axis, axis_size, vectors in iter_vector_views(tensor, embedding_dim):
                l2_stat = compute_l2_norm_stats_for_vectors(
                    name,
                    tensor_shape,
                    axis,
                    axis_size,
                    vectors,
                    histogram_dir,
                    histogram_bins,
                )
                if l2_stat is not None:
                    l2_rows.append(l2_stat)

                group_vectors["all_vectors"].append(vectors)
                lower_name = name.lower()
                if "wte" in lower_name and lower_name.endswith("weight"):
                    group_vectors["wte"].append(vectors)
                if ".attn.c_proj" in lower_name:
                    group_vectors["attn_c_proj"].append(vectors)
                if ".mlp.c_proj" in lower_name:
                    group_vectors["mlp_c_proj"].append(vectors)

                num_vectors = vectors.shape[0]
                if pairwise_limit > 0 and num_vectors > pairwise_limit:
                    console.print(
                        f"[yellow]Skipping pairwise statistics for {name} axis {axis} ({num_vectors:,} vectors) because it exceeds --pairwise-limit={pairwise_limit}.[/]"
                    )
                    continue

                histogram_prefix = None
                if histogram_prefix_root is not None:
                    histogram_prefix = f"{histogram_prefix_root}_{name.replace('/', '_')}"

                pairwise_rows.extend(
                    compute_pairwise_metrics(
                        name,
                        tensor_shape,
                        axis,
                        axis_size,
                        vectors,
                        histogram_dir,
                        histogram_bins,
                        histogram_prefix=histogram_prefix,
                        angle_units=angle_units,
                    )
                )

    if embedding_dim:
        combined_groups: Dict[str, List[torch.Tensor]] = {}
        if group_vectors["wte"]:
            combined_groups["wte"] = group_vectors["wte"]
        if group_vectors["attn_c_proj"]:
            combined_groups["attn_c_proj"] = group_vectors["attn_c_proj"]
        if group_vectors["mlp_c_proj"]:
            combined_groups["mlp_c_proj"] = group_vectors["mlp_c_proj"]
        if group_vectors["attn_c_proj"] or group_vectors["mlp_c_proj"]:
            combined_groups["all_c_proj"] = (
                group_vectors["attn_c_proj"] + group_vectors["mlp_c_proj"]
            )
        if group_vectors["wte"] or (group_vectors["attn_c_proj"] or group_vectors["mlp_c_proj"]):
            combined_groups["c_proj_plus_wte"] = (
                group_vectors["wte"]
                + group_vectors["attn_c_proj"]
                + group_vectors["mlp_c_proj"]
            )
        if group_vectors["all_vectors"]:
            combined_groups["all_vectors"] = group_vectors["all_vectors"]

        for group_name, tensors in combined_groups.items():
            if not tensors:
                continue
            combined = torch.cat(tensors, dim=0)
            num_vectors = combined.shape[0]
            if num_vectors <= 1:
                continue
            if pairwise_limit > 0 and num_vectors > pairwise_limit:
                console.print(
                    f"[yellow]Skipping pairwise statistics for group '{group_name}' ({num_vectors:,} vectors) because it exceeds --pairwise-limit={pairwise_limit}.[/]"
                )
                continue

            histogram_prefix = None
            if histogram_prefix_root is not None:
                histogram_prefix = f"{histogram_prefix_root}_group_{group_name}"

            group_pairwise_rows.extend(
                compute_pairwise_metrics(
                    f"[Group] {group_name}",
                    (combined.shape[0], combined.shape[1]),
                    -1,
                    combined.shape[1],
                    combined,
                    histogram_dir,
                    histogram_bins,
                    histogram_prefix=histogram_prefix,
                    angle_units=angle_units,
                )
            )

    rows.sort(key=lambda item: item[0])

    return CheckpointAnalysis(
        path=path,
        rows=rows,
        summary=summary,
        l2_rows=l2_rows,
        pairwise_rows=pairwise_rows,
        group_pairwise_rows=group_pairwise_rows,
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


def render_angle_difference_table(
    rows: List[AngleDifferenceStats],
    max_rows: Optional[int],
    *,
    colorize: bool,
    angle_units: str,
    title: str,
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
        ),
        ColumnSpec(
            "Q1",
            extractor=lambda row: float(row.q1),
            formatter=lambda row: f"{row.q1:.6g}",
        ),
        ColumnSpec(
            "Median",
            extractor=lambda row: float(row.median),
            formatter=lambda row: f"{row.median:.6g}",
        ),
        ColumnSpec(
            "Q3",
            extractor=lambda row: float(row.q3),
            formatter=lambda row: f"{row.q3:.6g}",
        ),
        ColumnSpec(
            "Q95",
            extractor=lambda row: float(row.q95),
            formatter=lambda row: f"{row.q95:.6g}",
        ),
    ]

    column_ranges = _compute_column_ranges(rows, column_specs)

    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        header_style="bold red",
        show_lines=False,
    )
    table.add_column("Parameter", overflow="fold")
    table.add_column("Shape", justify="center")
    table.add_column("Axis", justify="center")
    table.add_column("Units", justify="center")
    table.add_column("# Vectors", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Q05", justify="right")
    table.add_column("Q1", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Q3", justify="right")
    table.add_column("Q95", justify="right")
    if has_histograms:
        table.add_column("Histogram", overflow="fold")

    for row in display_rows:
        formatted_cells = [
            _format_with_color(spec, row, column_ranges, colorize=colorize)
            for spec in column_specs
        ]
        axis_label = (
            f"axis {row.axis} (vector dim={row.axis_size})"
            if row.axis >= 0
            else f"group (vector dim={row.axis_size})"
        )
        values = [
            row.parameter,
            str(row.tensor_shape),
            axis_label,
            angle_units,
            *formatted_cells,
        ]
        if has_histograms:
            values.append(str(row.histogram_path) if row.histogram_path else "—")
        table.add_row(*values)

    console = Console()
    console.print(table)
    if max_rows is not None and total_rows > max_rows:
        console.print(
            f"[dim]Displayed {len(display_rows)} of {total_rows} rows (limited by --max-angle-rows). Use a larger value to see more.[/]"
        )


def render_angle_overall_summary(
    summary: Dict[str, Any],
    *,
    angle_units: str,
) -> None:
    table = Table(title="Angular difference summary", box=box.SQUARE)
    table.add_column("Metric", style="bold red")
    table.add_column("Value", style="bold cyan", justify="right")

    total_vectors = summary.get("total_vectors", 0)
    valid_vectors = summary.get("valid_vectors", 0)
    table.add_row("Total vectors", f"{int(total_vectors):,}")
    table.add_row("Valid vectors", f"{int(valid_vectors):,}")
    if total_vectors:
        coverage = (valid_vectors / total_vectors) * 100
        table.add_row("Valid coverage", f"{coverage:.3g}%")
    stats = summary.get("stats", {})
    if stats:
        table.add_row(f"Min ({angle_units})", f"{stats.get('min', float('nan')):.6g}")
        table.add_row(f"Max ({angle_units})", f"{stats.get('max', float('nan')):.6g}")
        table.add_row(f"Mean ({angle_units})", f"{stats.get('mean', float('nan')):.6g}")
        table.add_row(f"Std ({angle_units})", f"{stats.get('std', float('nan')):.6g}")
        table.add_row(f"Q05 ({angle_units})", f"{stats.get('q05', float('nan')):.6g}")
        table.add_row(f"Q1 ({angle_units})", f"{stats.get('q1', float('nan')):.6g}")
        table.add_row(f"Median ({angle_units})", f"{stats.get('median', float('nan')):.6g}")
        table.add_row(f"Q3 ({angle_units})", f"{stats.get('q3', float('nan')):.6g}")
        table.add_row(f"Q95 ({angle_units})", f"{stats.get('q95', float('nan')):.6g}")
    histogram_path = summary.get("histogram_path")
    if histogram_path:
        table.add_row("Histogram", str(histogram_path))

    Console().print(table)


def write_angle_csv(
    csv_path: Path,
    records: List[Dict[str, Any]],
    *,
    angle_units: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parameter",
        "tensor_shape",
        "axis",
        "vector_index",
        f"angle_{angle_units}",
        "cosine_similarity",
        "l2_norm_ckpt1",
        "l2_norm_ckpt2",
        "valid",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def compare_checkpoint_angles(
    ckpt_a: Path,
    ckpt_b: Path,
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    pattern: re.Pattern,
    *,
    embedding_dim: Optional[int],
    angle_units: str,
    histogram_dir: Optional[Path],
    histogram_bins: int,
    histogram_prefix_root: Optional[str] = None,
) -> Tuple[List[AngleDifferenceStats], Dict[str, Any], List[Dict[str, Any]]]:
    console = Console()
    if embedding_dim is None:
        console.print(
            "[yellow]Embedding dimension unavailable; skipping angular difference comparison.[/]"
        )
        return [], {"total_vectors": 0, "valid_vectors": 0, "stats": {}}, []

    angle_rows: List[AngleDifferenceStats] = []
    overall_values: List[torch.Tensor] = []
    vector_records: List[Dict[str, Any]] = []
    total_vectors = 0
    valid_vectors = 0

    for name, tensor_a in state_dict_a.items():
        if not pattern.search(name):
            continue
        tensor_b = state_dict_b.get(name)
        if tensor_b is None:
            console.print(
                f"[yellow]Skipping {name}: not found in comparison checkpoint {ckpt_b}.[/]"
            )
            continue
        if tensor_a.shape != tensor_b.shape:
            console.print(
                f"[yellow]Skipping {name}: shape mismatch ({tensor_a.shape} vs {tensor_b.shape}).[/]"
            )
            continue

        tensor_shape = tuple(tensor_a.shape)
        for axis, axis_size, vectors_a in iter_vector_views(tensor_a, embedding_dim):
            moved_b = tensor_b.movedim(axis, -1)
            vectors_b = moved_b.reshape(-1, axis_size)
            if vectors_b.shape != vectors_a.shape:
                console.print(
                    f"[yellow]Skipping {name} axis {axis}: incompatible vector shapes between checkpoints.[/]"
                )
                continue

            angles, cosines, norms_a, norms_b, valid_mask = compute_vector_angle_metrics(
                vectors_a, vectors_b, angle_units=angle_units
            )
            num_vectors = angles.shape[0]
            total_vectors += num_vectors
            valid_vectors += int(valid_mask.sum().item())

            summary = summarize_metric_values(angles)
            finite_angles = angles[torch.isfinite(angles)]
            histogram_path: Optional[Path] = None
            if histogram_dir is not None and finite_angles.numel() > 0:
                histogram_prefix = None
                if histogram_prefix_root is not None:
                    histogram_prefix = f"{histogram_prefix_root}_{name.replace('/', '_')}"
                histogram_path = save_angle_histogram(
                    finite_angles,
                    histogram_dir,
                    name,
                    axis,
                    tensor_shape=tensor_shape,
                    axis_size=axis_size,
                    angle_units=angle_units,
                    bins=histogram_bins,
                    histogram_prefix=histogram_prefix,
                )

            angle_rows.append(
                AngleDifferenceStats(
                    parameter=name,
                    axis=axis,
                    axis_size=axis_size,
                    tensor_shape=tensor_shape,
                    num_vectors=num_vectors,
                    min=summary["min"],
                    max=summary["max"],
                    mean=summary["mean"],
                    std=summary["std"],
                    q05=summary["q05"],
                    q1=summary["q1"],
                    median=summary["median"],
                    q3=summary["q3"],
                    q95=summary["q95"],
                    histogram_path=histogram_path,
                )
            )

            if finite_angles.numel() > 0:
                overall_values.append(finite_angles)

            angles_list = angles.cpu().tolist()
            cos_list = cosines.cpu().tolist()
            norms_a_list = norms_a.cpu().tolist()
            norms_b_list = norms_b.cpu().tolist()
            valid_list = valid_mask.cpu().tolist()
            for idx, angle_val in enumerate(angles_list):
                record = {
                    "parameter": name,
                    "tensor_shape": str(tensor_shape),
                    "axis": axis,
                    "vector_index": idx,
                    f"angle_{angle_units}": angle_val,
                    "cosine_similarity": cos_list[idx],
                    "l2_norm_ckpt1": norms_a_list[idx],
                    "l2_norm_ckpt2": norms_b_list[idx],
                    "valid": int(valid_list[idx]),
                }
                vector_records.append(record)

    overall_summary: Dict[str, Any] = {
        "total_vectors": total_vectors,
        "valid_vectors": valid_vectors,
        "stats": {},
    }
    if overall_values:
        combined = torch.cat(overall_values)
        overall_summary["stats"] = summarize_metric_values(combined)
        if histogram_dir is not None and combined.numel() > 0:
            histogram_prefix = None
            if histogram_prefix_root is not None:
                histogram_prefix = f"{histogram_prefix_root}_overall"
            overall_summary["histogram_path"] = save_angle_histogram(
                combined,
                histogram_dir,
                f"{ckpt_a.stem}_vs_{ckpt_b.stem}",
                -1,
                tensor_shape=tuple(),
                axis_size=embedding_dim,
                angle_units=angle_units,
                bins=histogram_bins,
                histogram_prefix=histogram_prefix,
            )

    return angle_rows, overall_summary, vector_records


def main() -> None:
    args = parse_args()
    console = Console()

    ckpt_paths = [Path(path) for path in args.ckpt_paths]
    if len(ckpt_paths) == 0 or len(ckpt_paths) > 2:
        raise SystemExit("Provide one or two checkpoint paths for analysis.")

    state_dicts: List[Dict[str, torch.Tensor]] = []
    configs: List[GPTConfig] = []
    for path in ckpt_paths:
        try:
            state_dict, config = load_state_dict(str(path), args.device)
        except FileNotFoundError as exc:
            raise SystemExit(f"Checkpoint not found: {exc}") from exc
        state_dicts.append(state_dict)
        configs.append(config)

    pattern = re.compile(args.pattern)

    embedding_dim = args.embedding_dim
    if embedding_dim is None and configs:
        embedding_dim = getattr(configs[0], "n_embd", None)
        if embedding_dim is None:
            console.print(
                "[yellow]Embedding dimension not provided and not found in checkpoint config; directional statistics disabled.[/]"
            )
    if embedding_dim is not None and len(configs) == 2 and args.embedding_dim is None:
        other_dim = getattr(configs[1], "n_embd", None)
        if other_dim is not None and other_dim != embedding_dim:
            console.print(
                f"[yellow]Warning: checkpoint embedding dimensions differ ({embedding_dim} vs {other_dim}); using {embedding_dim} for analysis.[/]"
            )

    histogram_dir = args.histogram_dir
    if histogram_dir is not None:
        histogram_dir = histogram_dir.expanduser().resolve()
        histogram_dir.mkdir(parents=True, exist_ok=True)

    pairwise_limit = args.pairwise_limit if args.pairwise_limit is not None else 0
    analyses: List[CheckpointAnalysis] = []
    for idx, (path, state_dict) in enumerate(zip(ckpt_paths, state_dicts), start=1):
        histogram_prefix_root = None
        if histogram_dir is not None:
            prefix_base = path.stem
            if len(ckpt_paths) > 1:
                prefix_base = f"ckpt{idx}_{prefix_base}"
            histogram_prefix_root = prefix_base

        analysis = analyze_checkpoint(
            path,
            state_dict,
            pattern,
            embedding_dim=embedding_dim,
            histogram_dir=histogram_dir,
            histogram_bins=args.histogram_bins,
            pairwise_limit=pairwise_limit,
            angle_units=args.angle_units,
            histogram_prefix_root=histogram_prefix_root,
        )
        analyses.append(analysis)

    any_rows = any(analysis.rows for analysis in analyses)
    if not any_rows:
        console.print("[red]No parameters matched the provided pattern.[/]")
        return

    for idx, analysis in enumerate(analyses, start=1):
        console.print(
            f"\n[bold underline]Checkpoint {idx}: {analysis.path}[/]"
        )
        if not analysis.rows:
            console.print(
                f"[yellow]No parameters matched the provided pattern for checkpoint {analysis.path}.[/]"
            )
            continue
        render_table(analysis.rows, args.max_rows, colorize=args.colorize)
        render_summary(analysis.summary, matched=len(analysis.rows))
        if analysis.l2_rows:
            render_l2_table(analysis.l2_rows, args.max_l2_rows, colorize=args.colorize)
        if analysis.pairwise_rows:
            render_pairwise_table(
                analysis.pairwise_rows,
                args.max_pairwise_rows,
                colorize=args.colorize,
                title=f"Pairwise similarity statistics (per tensor) — {analysis.path.name}",
            )
        if analysis.group_pairwise_rows:
            render_pairwise_table(
                analysis.group_pairwise_rows,
                args.max_pairwise_rows,
                colorize=args.colorize,
                title=f"Pairwise similarity statistics (groups) — {analysis.path.name}",
            )

    if len(ckpt_paths) == 2:
        histogram_prefix_root = None
        if histogram_dir is not None:
            histogram_prefix_root = f"{ckpt_paths[0].stem}_vs_{ckpt_paths[1].stem}"
        angle_rows, overall_summary, vector_records = compare_checkpoint_angles(
            ckpt_paths[0],
            ckpt_paths[1],
            state_dicts[0],
            state_dicts[1],
            pattern,
            embedding_dim=embedding_dim,
            angle_units=args.angle_units,
            histogram_dir=histogram_dir,
            histogram_bins=args.histogram_bins,
            histogram_prefix_root=histogram_prefix_root,
        )

        if angle_rows:
            console.print(
                f"\n[bold underline]Angular comparisons: {ckpt_paths[0].name} vs {ckpt_paths[1].name}[/]"
            )
            render_angle_difference_table(
                angle_rows,
                args.max_angle_rows,
                colorize=args.colorize,
                angle_units=args.angle_units,
                title="Per-tensor angular difference statistics",
            )
            render_angle_overall_summary(
                overall_summary,
                angle_units=args.angle_units,
            )
            if vector_records:
                write_angle_csv(args.angle_csv, vector_records, angle_units=args.angle_units)
                console.print(
                    f"[green]Per-vector angular differences written to {args.angle_csv}[/]"
                )
        else:
            console.print(
                "[yellow]No overlapping parameters found for angular comparison or embedding dimension unavailable.[/]"
            )


if __name__ == "__main__":
    main()

