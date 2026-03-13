from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def decode_numerical_series(token_ids: List[int], meta: Dict[str, object], model_input_format: str) -> np.ndarray:
    """Decode token ids into numeric values aligned with model input format.

    For scalar integer-quantized datasets, if quantization metadata exists we
    approximately invert it for plotting.
    """
    input_format = model_input_format or str(meta.get("numerical_multicontext_input_format", "scalar"))

    if input_format == "fp16_bits":
        raw = np.asarray(token_ids, dtype=np.uint16)
        return raw.view(np.float16).astype(np.float32)

    values = np.asarray(token_ids, dtype=np.float32)
    quant = meta.get("quantization") if isinstance(meta, dict) else None
    if isinstance(quant, dict) and quant.get("type") == "shift_scale_round_clip_uint16":
        scale = float(quant.get("scale", 1.0))
        shift = float(quant.get("shift", 0.0))
        if scale != 0.0:
            values = (values / scale) - shift

    return values


def write_plotly_report(
    *,
    output_path: str,
    sample_series: List[Dict[str, List[float]]],
    prompt_lengths: Dict[str, int],
    include_prompt: bool,
) -> Path:
    """Write a single html page with per-channel traces across samples."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for --numerical_multicontext_plotly. Install with `pip install plotly`."
        ) from exc

    if not sample_series:
        raise ValueError("No sample series provided for plotly report")

    channel_names = list(sample_series[0].keys())
    fig = make_subplots(
        rows=len(channel_names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=channel_names,
    )

    for row_idx, channel_name in enumerate(channel_names, start=1):
        prompt_len = prompt_lengths.get(channel_name, 0)
        for sample_idx, sample_data in enumerate(sample_series, start=1):
            values = np.asarray(sample_data[channel_name], dtype=np.float32)
            if not include_prompt and prompt_len > 0:
                values = values[prompt_len:]
                x_vals = np.arange(prompt_len, prompt_len + len(values), dtype=np.int32)
            else:
                x_vals = np.arange(len(values), dtype=np.int32)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=values,
                    mode="lines",
                    name=f"sample_{sample_idx}",
                    legendgroup=f"sample_{sample_idx}",
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )

        if include_prompt and prompt_len > 0:
            fig.add_vline(
                x=prompt_len - 1,
                line_width=1,
                line_dash="dash",
                line_color="gray",
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(title_text=channel_name, row=row_idx, col=1)

    fig.update_xaxes(title_text="time step", row=len(channel_names), col=1)
    fig.update_layout(
        title="Numerical Multicontext Samples",
        height=max(350 * len(channel_names), 500),
        template="plotly_white",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    return output
