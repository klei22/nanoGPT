"""Compare naive fake W4A4 with group-wise regular-Hadamard ConvRot."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

# Allow both ``python -m quantizations.ptq.convrot_demo`` and the repo's usual
# ``python quantizations/ptq/<script>.py`` invocation style.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantizations.ptq.convrot import fake_w4a4_linear, group_rotate, regular_hadamard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="nanoGPT ckpt.pt to inspect")
    parser.add_argument("--tensor", help="Regex selecting a two-dimensional weight")
    parser.add_argument("--group-size", type=int, default=16, help="RHT order (4**k)")
    parser.add_argument("--tokens", type=int, default=32, help="Synthetic activation rows")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--row-outlier", type=float, default=12.0,
                        help="Common-mode value added to the first activation row")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def select_weight(state: dict, pattern: str | None, group_size: int) -> tuple[str, torch.Tensor]:
    regex = re.compile(pattern) if pattern else None
    for name, value in state.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and value.shape[1] % group_size == 0
            and (regex is None or regex.search(name))
        ):
            return name, value.detach().float().cpu()
    raise ValueError("no matching 2-D weight has an input dimension divisible by group size")


def relative_rmse(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((actual - reference) ** 2)) / reference.square().mean().sqrt())


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    name, weight = select_weight(state, args.tensor, args.group_size)

    generator = torch.Generator().manual_seed(args.seed)
    activations = torch.randn(args.tokens, weight.shape[1], generator=generator)
    activations[0].add_(args.row_outlier)
    rotation = regular_hadamard(args.group_size, dtype=weight.dtype)
    reference = activations @ weight.T
    rotated_reference = group_rotate(activations, rotation) @ group_rotate(weight, rotation).T
    naive = fake_w4a4_linear(activations, weight)
    convrot = fake_w4a4_linear(activations, weight, rotation=rotation)

    report = {
        "tensor": name,
        "shape": list(weight.shape),
        "group_size": args.group_size,
        "equivalence_max_abs_error": float((reference - rotated_reference).abs().max()),
        "activation_peak_before": float(activations.abs().max()),
        "activation_peak_after_rotation": float(group_rotate(activations, rotation).abs().max()),
        "naive_w4a4_relative_rmse": relative_rmse(naive, reference),
        "convrot_w4a4_relative_rmse": relative_rmse(convrot, reference),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
