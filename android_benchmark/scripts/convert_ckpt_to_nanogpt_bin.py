#!/usr/bin/env python3
"""Convert a compatible nanoGPT ckpt.pt to the tiny Android benchmark format.

The JNI benchmark intentionally supports the repo's default GPT path only:
RMSNorm, absolute positional embeddings, causal attention, GELU MLP, no bias in
attention projections, and an optional bias on the MLP up projection. The script
fails fast if a checkpoint uses unsupported architectural variations so benchmark
numbers are not silently wrong.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

from typing import Any


MAGIC = 0x4E475054  # NGPT
VERSION = 1


def require(args: dict, key: str, expected):
    actual = args.get(key, expected)
    if actual != expected:
        raise SystemExit(f"Unsupported checkpoint option {key}={actual!r}; expected {expected!r}.")


def tensor(sd: dict, name: str) -> Any:
    if name not in sd:
        raise SystemExit(f"Missing tensor {name!r}; this checkpoint is not supported by the Android benchmark exporter.")
    return sd[name].detach().cpu().float().contiguous()


def write_tensor(f, t: Any) -> None:
    flat = t.detach().cpu().float().reshape(-1).contiguous()
    f.write(struct.pack("<i", flat.numel()))
    f.write(flat.numpy().astype("<f4", copy=False).tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt", type=Path, help="Path to a repo training output checkpoint, e.g. out/run/ckpt.pt")
    p.add_argument("--out", type=Path, default=Path("android_benchmark/app/src/main/assets/nanogpt.bin"))
    p.add_argument("--allow-untied-head", action="store_true", help="Use lm_head.weight when it differs from transformer.wte.weight.")
    args = p.parse_args()

    import torch

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model_args = dict(ckpt.get("model_args", {}))

    for key, expected in {
        "attention_variant": "causal",
        "mlp_variant": "mlp",
        "norm_variant_attn": "rmsnorm",
        "norm_variant_output": "rmsnorm",
        "use_abs_pos_embeddings": True,
        "use_rotary_embeddings": False,
        "bias": False,
        "n_embd_wte": None,
        "multicontext": False,
        "use_moe": False,
        "gate": False,
        "use_parallel_mlp": False,
        "use_edgellm_asic": False,
    }.items():
        require(model_args, key, expected)

    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    n_embd = int(model_args["n_embd"])
    block_size = int(model_args["block_size"])
    vocab_size = int(model_args["vocab_size"])
    hidden = int(model_args.get("mlp_size") or model_args.get("mlp_expansion_factor", 4) * n_embd)

    wte = tensor(sd, "transformer.wte.weight")
    lm_head = sd.get("lm_head.weight", wte).detach().cpu().float().contiguous()
    if not torch.equal(lm_head, wte) and not args.allow_untied_head:
        raise SystemExit("lm_head.weight differs from transformer.wte.weight; pass --allow-untied-head if intentional.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(struct.pack("<8I", MAGIC, VERSION, block_size, vocab_size, n_layer, n_head, n_embd, hidden))
        write_tensor(f, wte)
        write_tensor(f, tensor(sd, "transformer.wpe.embedding.weight"))
        for i in range(n_layer):
            prefix = f"transformer.h.{i}"
            write_tensor(f, tensor(sd, f"{prefix}.pre_ln_attn.gain"))
            write_tensor(f, tensor(sd, f"{prefix}.pre_ln_mlp.gain"))
            write_tensor(f, tensor(sd, f"{prefix}.attn.c_attn_q.weight"))
            write_tensor(f, tensor(sd, f"{prefix}.attn.c_attn_k.weight"))
            write_tensor(f, tensor(sd, f"{prefix}.attn.c_attn_v.weight"))
            write_tensor(f, tensor(sd, f"{prefix}.attn.c_proj.weight"))
            write_tensor(f, tensor(sd, f"{prefix}.mlp.c_fc.weight"))
            fc_bias = sd.get(f"{prefix}.mlp.c_fc.bias")
            write_tensor(f, torch.zeros(hidden) if fc_bias is None else fc_bias)
            write_tensor(f, tensor(sd, f"{prefix}.mlp.c_proj.weight"))
        write_tensor(f, tensor(sd, "transformer.ln_f.gain"))
        write_tensor(f, lm_head)

    print(f"Wrote {args.out} ({args.out.stat().st_size / (1024 * 1024):.2f} MiB)")


if __name__ == "__main__":
    main()
