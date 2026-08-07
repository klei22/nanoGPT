"""Fine-tune only the smallest per-channel RMSNorm gains toward zero."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


Selection = Dict[str, torch.Tensor]


def find_rmsnorm_weights(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    """Return one-dimensional gain parameters belonging to RMSNorm modules."""
    weights = {}
    for module_name, module in model.named_modules():
        if "rmsnorm" not in module.__class__.__name__.lower():
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter) and weight.ndim == 1:
            name = f"{module_name}.weight" if module_name else "weight"
            weights[name] = weight
    if not weights:
        raise ValueError("No one-dimensional RMSNorm weight parameters were found")
    return weights


def select_smallest_channels(
    weights: Dict[str, torch.nn.Parameter], fraction: float
) -> Selection:
    """Select at least one lowest-absolute-value channel in each RMSNorm."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("selection fraction must be in (0, 1]")
    selection = {}
    for name, weight in weights.items():
        count = max(1, math.ceil(weight.numel() * fraction))
        indices = torch.topk(weight.detach().abs(), count, largest=False).indices
        mask = torch.zeros_like(weight, dtype=torch.bool)
        mask[indices] = True
        selection[name] = mask
    return selection


def configure_selected_gradients(
    model: torch.nn.Module,
    weights: Dict[str, torch.nn.Parameter],
    selection: Selection,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Freeze the model and mask RMSNorm gradients to selected entries."""
    model.requires_grad_(False)
    handles = []
    for name, weight in weights.items():
        weight.requires_grad_(True)
        mask = selection[name]
        handles.append(weight.register_hook(lambda grad, mask=mask: grad * mask))
    return handles


def restore_unselected(
    weights: Dict[str, torch.nn.Parameter],
    initial: Dict[str, torch.Tensor],
    selection: Selection,
) -> None:
    """Restore frozen entries after an optimizer step (including weight decay)."""
    with torch.no_grad():
        for name, weight in weights.items():
            weight[~selection[name]] = initial[name][~selection[name]]


def selected_l1(weights: Dict[str, torch.nn.Parameter], selection: Selection) -> torch.Tensor:
    values = [weight[selection[name]].abs() for name, weight in weights.items()]
    return torch.cat(values).mean()


def threshold_selected(
    weights: Dict[str, torch.nn.Parameter], selection: Selection, threshold: float
) -> int:
    """Set selected gains at or below the absolute threshold exactly to zero."""
    if threshold < 0:
        raise ValueError("zero threshold must be non-negative")
    zeroed = 0
    with torch.no_grad():
        for name, weight in weights.items():
            to_zero = selection[name] & (weight.abs() <= threshold)
            zeroed += int(to_zero.sum().item())
            weight[to_zero] = 0
    return zeroed


def selection_manifest(
    weights: Dict[str, torch.nn.Parameter], selection: Selection
) -> Dict[str, object]:
    layers = {}
    for name, weight in weights.items():
        indices = selection[name].nonzero(as_tuple=False).flatten().cpu().tolist()
        layers[name] = {
            "channels": indices,
            "initial_values": weight.detach().cpu()[selection[name].cpu()].tolist(),
            "total_channels": weight.numel(),
        }
    return {"selected_channels": sum(len(v["channels"]) for v in layers.values()), "layers": layers}


def tokenize_stream(
    rows: Iterable[dict], tokenizer, columns: List[str], sequence_length: int
) -> Iterable[torch.Tensor]:
    """Pack streaming text rows into fixed-length token sequences."""
    buffer: List[int] = []
    separator = tokenizer.eos_token_id
    for row in rows:
        text = "\n".join(str(row[column]) for column in columns if row.get(column))
        if not text:
            continue
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if separator is not None:
            buffer.append(separator)
        while len(buffer) >= sequence_length:
            yield torch.tensor(buffer[:sequence_length], dtype=torch.long)
            del buffer[:sequence_length]


def evaluate(model, batches: List[torch.Tensor], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            ids = batch.unsqueeze(0).to(device)
            losses.append(float(model(input_ids=ids, labels=ids).loss))
    return sum(losses) / len(losses) if losses else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--extra-text-column", action="append", default=[])
    parser.add_argument("--selection-fraction", type=float, default=0.05)
    parser.add_argument("--zero-penalty", type=float, default=0.1)
    parser.add_argument("--zero-threshold", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="outputs/rmsnorm-channel-zeroing")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.sequence_length < 2 or args.eval_batches < 1:
        raise ValueError("max-steps and eval-batches must be positive; sequence-length must be at least 2")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weights = find_rmsnorm_weights(model)
    selection = select_smallest_channels(weights, args.selection_fraction)
    manifest = selection_manifest(weights, selection)
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    initial = {name: weight.detach().clone() for name, weight in weights.items()}
    handles = configure_selected_gradients(model, weights, selection)

    dataset_args = [args.dataset]
    if args.dataset_config:
        dataset_args.append(args.dataset_config)
    rows = load_dataset(*dataset_args, split=args.dataset_split, streaming=True).shuffle(
        seed=args.seed, buffer_size=10_000
    )
    columns = [args.text_column, *args.extra_text_column]
    batches = iter(tokenize_stream(rows, tokenizer, columns, args.sequence_length))
    evaluation = [next(batches) for _ in range(args.eval_batches)]
    initial_eval_loss = evaluate(model, evaluation, device)

    optimizer = torch.optim.AdamW(
        list(weights.values()), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model.train()
    final_train_loss: Optional[float] = None
    for step in range(1, args.max_steps + 1):
        ids = next(batches).unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        lm_loss = model(input_ids=ids, labels=ids).loss
        loss = lm_loss + args.zero_penalty * selected_l1(weights, selection)
        loss.backward()
        optimizer.step()
        restore_unselected(weights, initial, selection)
        final_train_loss = float(loss.detach())
        if step == 1 or step % 25 == 0:
            print(f"step={step} loss={final_train_loss:.6f} lm_loss={float(lm_loss):.6f}")

    zeroed = threshold_selected(weights, selection, args.zero_threshold)
    final_eval_loss = evaluate(model, evaluation, device)
    for handle in handles:
        handle.remove()
    metrics = {
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "final_train_loss": final_train_loss,
        "selected_channels": manifest["selected_channels"],
        "zeroed_channels": zeroed,
    }
    (output_dir / "experiment_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
