#!/usr/bin/env python3
"""Train the existing GPT multicontext backbone and export native 3D trajectories.

No dataset downloads, second decoder, or separate per-head output norm. One
sequence position contains BOTH a digit and a letter. Loss = (CE_d + CE_l)/2.
"""

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpt_conf import GPTConfig
from model import GPT
from variations.small_circle_embeddings import SmallCircleEmbedding

VARIANTS = ("table_free", "table_sphere", "great_circle", "small_circle")
REFERENCE_COMMIT = "6cb931dcccada904e523a15a18898e609bde1fb0"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")),
                         encoding="utf-8")
    temporary.replace(path)


def make_config(args, variant):
    circle = variant in ("great_circle", "small_circle")
    return GPTConfig(
        n_embd=3, n_layer=1, n_head=args.heads, n_kv_group=args.heads,
        block_size=args.block_size, vocab_size=args.digit_slots,
        vocab_sizes=[args.digit_slots, args.letters], multicontext=True,
        dropout=0.0, bias=False, use_abs_pos_embeddings=False,
        use_rotary_embeddings=False, activation_variant="gelu",
        mlp_expansion_factor=args.mlp_expansion, norm_variant_attn="rmsnorm",
        norm_variant_output="rmsnorm", wte_weight_tying=True,
        wte_fixed_norm=variant != "table_free", wte_fixed_norm_value=args.radius,
        multicontext_embedding_variant="small_circle" if circle else "table",
        circle_offset_init=0.0 if variant == "great_circle" else args.circle_offset,
        circle_learn_offset=variant == "small_circle",
    )


def make_model(args, variant, seed):
    torch.manual_seed(seed)
    model = GPT(make_config(args, variant))
    # Start free/spherical tables at identical norms; only the latter is projected
    # after optimization. The usual Gaussian *directions* are retained.
    if variant == "table_free":
        model.reproject_token_embeddings()
    return model.to(args.device)


def batches(digit_starts, letter_starts, args):
    offsets = torch.arange(args.block_size + 1, device=args.device)
    d = (digit_starts[:, None] + offsets) % args.digits
    l = (letter_starts[:, None] + offsets) % args.letters
    inputs = {"digits": d[:, :-1].contiguous(), "letters": l[:, :-1].contiguous()}
    targets = {"digits": d[:, 1:].contiguous(), "letters": l[:, 1:].contiguous()}
    return inputs, targets


def evaluation_batch(args):
    if args.pairing == "aligned":
        starts = torch.arange(math.lcm(args.digits, args.letters), device=args.device)
        return batches(starts, starts, args)
    pairs = torch.cartesian_prod(torch.arange(args.digits), torch.arange(args.letters))
    return batches(pairs[:, 0].to(args.device), pairs[:, 1].to(args.device), args)


def training_batch(args, generator):
    if args.pairing == "aligned":
        d = torch.randint(math.lcm(args.digits, args.letters), (args.batch_size,), generator=generator)
        l = d
    else:
        d = torch.randint(args.digits, (args.batch_size,), generator=generator)
        l = torch.randint(args.letters, (args.batch_size,), generator=generator)
    return batches(d.to(args.device), l.to(args.device), args)


def geometry_json(module):
    if not isinstance(module, SmallCircleEmbedding):
        return None
    return {key: value.detach().cpu().tolist() for key, value in module.geometry().items()}


@torch.no_grad()
def snapshot(model, args, iteration, eval_batch):
    model.eval()
    inputs, targets = eval_batch
    logits, losses = model(None, token_dict=inputs, target_dict=targets, iter_num=iteration)
    predictions = [head.argmax(-1) for head in logits]
    correct = [predictions[i] == target for i, target in enumerate(targets.values())]
    tables = [model.transformer[f"wte_{i}"].weight.detach() for i in range(2)]
    norm = torch.cat(tables).norm(dim=-1)
    frame = {
        "iteration": iteration,
        "positions": torch.cat(tables).cpu().tolist(),
        "circles": [geometry_json(model.transformer[f"wte_{i}"]) for i in range(2)],
        "metrics": {
            "loss": float((losses[0] + losses[1]) / 2),
            "digit_loss": float(losses[0]), "letter_loss": float(losses[1]),
            "digit_accuracy": float(correct[0].float().mean()),
            "letter_accuracy": float(correct[1].float().mean()),
            "joint_accuracy": float((correct[0] & correct[1]).float().mean()),
            "norm_min": float(norm.min()), "norm_max": float(norm.max()),
        },
    }
    if not all(math.isfinite(value) for value in frame["metrics"].values()):
        raise FloatingPointError(f"nonfinite metrics at iteration {iteration}")
    return frame


@torch.no_grad()
def continuous_probe(model, args):
    """An explicit unseen-fractional-input probe; geometry is not generalization."""
    if not model.uses_circle_multicontext:
        return None
    device = args.device
    starts = torch.arange(math.lcm(args.digits, args.letters), device=device)
    fractions = torch.tensor([0.25, 0.5, 0.75], device=device)
    d_starts = (starts[:, None] + fractions).flatten()
    # Different fractional offsets for the two streams.
    l_starts = (starts[:, None] + fractions.flip(0)).flatten()
    inputs, targets = batches(d_starts, l_starts, args)
    inputs = {"digits": inputs["digits"] / args.digit_slots,
              "letters": inputs["letters"] / args.letters}
    hidden = []
    hook = model.transformer.ln_f.register_forward_hook(lambda _m, _i, output: hidden.append(output))
    try:
        model(None, token_dict=inputs)
    finally:
        hook.remove()
    result = {"description": "Unseen fractional input phases; no fractional targets were trained.",
              "examples": []}
    predicted, expected = [], []
    for i, (name, slots) in enumerate((("digits", args.digit_slots), ("letters", args.letters))):
        phase = model.transformer[f"wte_{i}"].decode_phase(hidden[0][:, -1])
        target = targets[name][:, -1] / slots
        error = (phase - target + 0.5).remainder(1) - 0.5
        result[f"{name}_circular_phase_mae"] = float(error.abs().mean())
        geometry = model.transformer[f"wte_{i}"].geometry()
        strength = torch.hypot(hidden[0][:, -1] @ geometry["u"], hidden[0][:, -1] @ geometry["v"])
        result[f"{name}_min_in_plane_projection"] = float(strength.min())
        predicted.append((phase * slots).cpu().tolist())
        expected.append((target * slots).cpu().tolist())
    for j in range(min(12, len(d_starts))):
        result["examples"].append({"expected": [expected[0][j], expected[1][j]],
                                   "predicted": [predicted[0][j], predicted[1][j]]})
    return result


def backbone_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_run(args, variant, seed, shared_backbone, matched_weights=None):
    name = f"{variant}-seed-{seed}"
    destination = args.output_dir / f"{name}.json"
    checkpoint = args.checkpoint_dir / f"{name}.pt"
    if (destination.exists() or checkpoint.exists()) and not (args.resume or args.overwrite):
        raise FileExistsError(f"{name} already exists; choose a new output directory, --resume, or --overwrite")
    model = make_model(args, variant, seed)
    model.load_state_dict(shared_backbone, strict=False)
    if matched_weights is not None and variant.startswith("table"):
        with torch.no_grad():
            for i in range(2):
                model.transformer[f"wte_{i}"].weight.copy_(matched_weights[i])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 10000)
    digit_tokens = [str(i) for i in range(args.digit_slots)]
    letter_tokens = list("abcdefghijklmnopqrstuvwxyz"[:args.letters])
    config = {key: value for key, value in vars(args).items()
              if key not in ("output_dir", "checkpoint_dir", "resume", "overwrite", "variants", "seeds")}
    payload = {
        "schema_version": 1, "task": "dual_stream_clock", "name": name,
        "variant": variant, "seed": seed, "config": config,
        "reference_commit": REFERENCE_COMMIT,
        "backbone_initial_sha256": backbone_hash(shared_backbone),
        "tokens": digit_tokens + letter_tokens,
        "trained_tokens": digit_tokens[:args.digits] + letter_tokens,
        "unseen_tokens": digit_tokens[args.digits:],
        "groups": [{"name": "digits", "start": 0, "size": args.digit_slots,
                    "active": args.digits},
                   {"name": "letters", "start": args.digit_slots, "size": args.letters,
                    "active": args.letters}],
        "projection": {"method": "native", "input_dimensions": 3},
        "fixed_norm": None if variant == "table_free" else args.radius,
        "wte_weight_tying": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "evaluation": "All aligned cycle starts" if args.pairing == "aligned" else "All digit/letter start pairs",
        "evaluation_note": "Repeated synthetic cycle evaluation, not held-out generalization.",
        "frames": [], "training": [], "completed": False,
    }
    first_step = 0
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if saved["variant"] != variant or saved["seed"] != seed:
            raise ValueError("checkpoint identity mismatch")
        old_config = saved["payload"]["config"].copy()
        new_config = config.copy()
        for key in ("steps", "checkpoint_every", "log_every", "threads", "device"):
            old_config.pop(key, None)
            new_config.pop(key, None)
        if old_config != new_config:
            raise ValueError("resume requires the same model, data, optimizer and snapshot settings")
        first_step = saved["iteration"]
        if first_step > args.steps:
            raise ValueError("requested steps precede checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["batch_rng"].cpu())
        payload = saved["payload"]
        payload["config"] = config
        payload["completed"] = False
    elif args.resume and destination.exists():
        raise FileNotFoundError(f"cannot resume {name} without {checkpoint}")
    eval_batch = evaluation_batch(args)
    if not payload["frames"]:
        payload["frames"].append(snapshot(model, args, 0, eval_batch))
    start_time = time.perf_counter()
    previous_seconds = payload.get("training_seconds", 0)

    def save(step):
        payload["training_seconds"] = previous_seconds + time.perf_counter() - start_time
        payload["timesteps_trained"] = step * args.batch_size * args.block_size
        payload["channel_tokens_trained"] = 2 * payload["timesteps_trained"]
        atomic_json(destination, payload)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "model_args": asdict(model.config),
                    "optimizer": optimizer.state_dict(), "batch_rng": generator.get_state(),
                    "iteration": step, "variant": variant, "seed": seed, "payload": payload}, temporary)
        temporary.replace(checkpoint)

    for step in range(first_step + 1, args.steps + 1):
        model.train()
        inputs, targets = training_batch(args, generator)
        optimizer.zero_grad(set_to_none=True)
        _, losses = model(None, token_dict=inputs, target_dict=targets, iter_num=step - 1)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss in {name} at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if model.config.wte_fixed_norm:
            model.reproject_token_embeddings()
        payload["training"].append({"update": step, "pre_update_loss": float(loss.detach())})
        if step % args.snapshot_every == 0 or step == args.steps:
            payload["frames"].append(snapshot(model, args, step, eval_batch))
        if step % args.log_every == 0 or step == args.steps:
            m = payload["frames"][-1]["metrics"]
            print(f"{name} {step}/{args.steps}: eval={m['loss']:.4f}, "
                  f"digits={m['digit_accuracy']:.1%}, letters={m['letter_accuracy']:.1%}", flush=True)
        if step % args.checkpoint_every == 0:
            save(step)
    model.eval()
    payload["continuous_probe"] = continuous_probe(model, args)
    payload["completed"] = True
    save(args.steps)
    return payload


def write_manifest(output_dir):
    runs = []
    for path in sorted(output_dir.glob("*-seed-*.json")):
        value = json.loads(path.read_text())
        if not value.get("completed"):
            continue
        runs.append({"name": value["name"], "file": path.name, "variant": value["variant"],
                     "seed": value["seed"], "config": value["config"],
                     "parameter_count": value["parameter_count"],
                     "final": value["frames"][-1]["metrics"]})
    atomic_json(output_dir / "manifest.json", {"schema_version": 1, "runs": runs})
    # Do not aggregate unlike experimental settings into one result.
    groups = {}
    for run in runs:
        key = (run["variant"], json.dumps(run["config"], sort_keys=True))
        groups.setdefault(key, []).append(run)
    summary = []
    for (variant, config), items in groups.items():
        row = {"variant": variant, "config": json.loads(config), "seeds": [r["seed"] for r in items]}
        for metric in ("loss", "digit_accuracy", "letter_accuracy", "joint_accuracy"):
            values = [r["final"][metric] for r in items]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    atomic_json(output_dir / "summary.json", summary)
    if runs:
        with (output_dir / "summary.csv").open("w", newline="") as handle:
            fields = ["name", "variant", "seed", "parameter_count"] + list(runs[0]["final"])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for run in runs:
                writer.writerow({**{key: run[key] for key in fields[:4]}, **run["final"]})
    return runs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--digits", type=int, default=8, help="Number of active digit targets (0 through N-1)")
    parser.add_argument("--digit-slots", type=int, default=10, help="Numeric vocabulary and clock size; 8 and 9 are untargeted by default")
    parser.add_argument("--letters", type=int, default=5)
    parser.add_argument("--pairing", choices=["aligned", "independent"], default="aligned")
    parser.add_argument("--heads", type=int, choices=[1, 3], default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--radius", type=float, default=math.sqrt(3))
    parser.add_argument("--circle-offset", type=float, default=0.5)
    parser.add_argument("--embedding-init", choices=["random", "matched_circle"], default="random")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/threejs/digits-3d/dual-stream")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "out/dual_stream_clock")
    conflict = parser.add_mutually_exclusive_group()
    conflict.add_argument("--resume", action="store_true")
    conflict.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= args.digits <= args.digit_slots <= 10 or not 2 <= args.letters <= 26:
        parser.error("require 2 <= digits <= digit-slots <= 10 and 2 <= letters <= 26")
    for key in ("steps", "block_size", "batch_size", "mlp_expansion", "snapshot_every", "checkpoint_every", "log_every", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if not 0 < args.circle_offset < 0.9999:
        parser.error("circle-offset must be in (0, 0.9999)")
    if not all(math.isfinite(v) and v > 0 for v in (args.radius, args.learning_rate, args.grad_clip)):
        parser.error("radius, learning-rate and grad-clip must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        reference = make_model(args, "table_sphere", seed)
        backbone = {key: value.detach().clone() for key, value in reference.state_dict().items()
                    if key.startswith(("transformer.h.", "transformer.ln_f."))}
        matched = None
        if args.embedding_init == "matched_circle":
            circle = make_model(args, "small_circle", seed)
            matched = [circle.transformer[f"wte_{i}"].weight.detach().clone() for i in range(2)]
        for variant in dict.fromkeys(args.variants):
            train_run(args, variant, seed, backbone, matched)
            write_manifest(args.output_dir)
    print(f"Completed. Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
