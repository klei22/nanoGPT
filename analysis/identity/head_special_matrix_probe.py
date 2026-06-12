#!/usr/bin/env python3
"""
head_special_matrix_probe.py

Only generates per-layer/per-head heatmaps for whether W_O^h W_V^h
resembles identity, inverse/rotation-like, projection-like, etc.

Examples:

  python head_special_matrix_probe.py --preset qwen-0.5b --device cuda

  python head_special_matrix_probe.py --preset gemma-3-270m --device cuda --dtype bfloat16

  python head_special_matrix_probe.py \
    --model google/gemma-3-270m \
    --device cuda \
    --dtype bfloat16 \
    --trust-remote-code
"""

import argparse
import os
import csv

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument(
        "--preset",
        default=None,
        choices=["qwen-0.5b", "gemma-3-270m"],
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--outdir", default="./head_special_matrix_out")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--dpi", type=int, default=200)

    return p.parse_args()


def apply_preset(args):
    if args.preset == "qwen-0.5b":
        args.model = "Qwen/Qwen2.5-0.5B"

    elif args.preset == "gemma-3-270m":
        args.model = "google/gemma-3-270m"
        args.trust_remote_code = True

    return args


def get_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def save_heatmap(arr, title, xlabel, ylabel, cbar, path, dpi=200, show=False):
    plt.figure(figsize=(14, 8))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label=cbar)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if arr.shape[1] <= 64:
        plt.xticks(range(arr.shape[1]))
    if arr.shape[0] <= 80:
        plt.yticks(range(arr.shape[0]))

    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def cosine_to(A, B):
    return F.cosine_similarity(A.flatten(), B.flatten(), dim=0).item()


def rel_frob(A, B, eps=1e-12):
    return (torch.linalg.norm(A - B) / (torch.linalg.norm(B) + eps)).item()


def frob(A, B):
    return torch.linalg.norm(A - B).item()


def infer_head_shapes(Wq, Wk, Wv, Wo, hidden, config_heads=None):
    if config_heads is not None and Wq.shape[0] % config_heads == 0:
        head_dim = Wq.shape[0] // config_heads
    else:
        raise ValueError("Could not infer head_dim from config_heads.")

    q_heads = Wq.shape[0] // head_dim
    k_heads = Wk.shape[0] // head_dim
    v_heads = Wv.shape[0] // head_dim
    o_heads = Wo.shape[1] // head_dim

    return head_dim, q_heads, k_heads, v_heads, o_heads


def main():
    args = apply_preset(parse_args())
    os.makedirs(args.outdir, exist_ok=True)

    dtype = get_dtype(args.dtype)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    print(f"Loading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)

    model.eval()

    cfg = model.config
    hidden = cfg.hidden_size
    config_heads = getattr(cfg, "num_attention_heads", None)
    config_kv_heads = getattr(cfg, "num_key_value_heads", None)

    print("hidden:", hidden)
    print("config num_attention_heads:", config_heads)
    print("config num_key_value_heads:", config_kv_heads)

    layers = model.model.layers
    if args.max_layers is not None:
        layers = layers[:args.max_layers]

    metric_names = [
        "cos_identity_OV",
        "rel_frob_identity_OV",
        "frob_identity_OV",

        "cos_negative_identity_OV",
        "rel_frob_negative_identity_OV",

        "rel_frob_orthogonal_OVtOV_I",
        "cos_orthogonal_OVtOV_I",

        "rel_frob_involution_OV2_I",
        "cos_involution_OV2_I",

        "rel_frob_projection_OV2_OV",
        "cos_projection_OV2_OV",

        "rel_frob_symmetric_OV_OVt",
        "cos_symmetric_OV_OVt",

        "rel_frob_skew_OV_negOVt",
        "cos_skew_OV_negOVt",

        "rel_frob_headspace_VO_I",
        "cos_headspace_VO_I",
    ]

    scores = {m: [] for m in metric_names}
    rows = []

    with torch.no_grad():
        for layer_idx, block in enumerate(layers):
            attn = block.self_attn

            Wq = attn.q_proj.weight.detach().float()
            Wk = attn.k_proj.weight.detach().float()
            Wv = attn.v_proj.weight.detach().float()
            Wo = attn.o_proj.weight.detach().float()

            head_dim, q_heads, k_heads, v_heads, o_heads = infer_head_shapes(
                Wq, Wk, Wv, Wo, hidden, config_heads
            )

            Wv_h = Wv.view(v_heads, head_dim, hidden)
            Wo_h = Wo.view(hidden, o_heads, head_dim).permute(1, 0, 2)

            usable_heads = min(q_heads, o_heads)
            kv_group_size = max(1, q_heads // max(1, v_heads))

            if layer_idx == 0:
                print("inferred head_dim:", head_dim)
                print("inferred q_heads:", q_heads)
                print("inferred k_heads:", k_heads)
                print("inferred v_heads:", v_heads)
                print("inferred o_heads:", o_heads)
                print("usable_heads:", usable_heads)
                print("kv_group_size:", kv_group_size)

            print(f"Layer {layer_idx}")

            I_hidden = torch.eye(hidden, dtype=torch.float32, device=Wv.device)
            neg_I_hidden = -I_hidden
            I_head = torch.eye(head_dim, dtype=torch.float32, device=Wv.device)

            layer_metric_scores = {m: [] for m in metric_names}

            for h in range(usable_heads):
                kv_h = min(h // kv_group_size, v_heads - 1)

                # Hidden-space operator:
                # OV: [hidden, hidden]
                OV = Wo_h[h] @ Wv_h[kv_h]

                # Head-space operator:
                # VO: [head_dim, head_dim]
                # If this is identity-like, then Wv and Wo are inverse-like
                # inside the small head subspace.
                VO = Wv_h[kv_h] @ Wo_h[h]

                OVt = OV.T
                OV2 = OV @ OV
                OVtOV = OVt @ OV

                values = {}

                values["cos_identity_OV"] = cosine_to(OV, I_hidden)
                values["rel_frob_identity_OV"] = rel_frob(OV, I_hidden)
                values["frob_identity_OV"] = frob(OV, I_hidden)

                values["cos_negative_identity_OV"] = cosine_to(OV, neg_I_hidden)
                values["rel_frob_negative_identity_OV"] = rel_frob(OV, neg_I_hidden)

                values["rel_frob_orthogonal_OVtOV_I"] = rel_frob(OVtOV, I_hidden)
                values["cos_orthogonal_OVtOV_I"] = cosine_to(OVtOV, I_hidden)

                values["rel_frob_involution_OV2_I"] = rel_frob(OV2, I_hidden)
                values["cos_involution_OV2_I"] = cosine_to(OV2, I_hidden)

                values["rel_frob_projection_OV2_OV"] = rel_frob(OV2, OV)
                values["cos_projection_OV2_OV"] = cosine_to(OV2, OV)

                values["rel_frob_symmetric_OV_OVt"] = rel_frob(OV, OVt)
                values["cos_symmetric_OV_OVt"] = cosine_to(OV, OVt)

                values["rel_frob_skew_OV_negOVt"] = rel_frob(OV, -OVt)
                values["cos_skew_OV_negOVt"] = cosine_to(OV, -OVt)

                values["rel_frob_headspace_VO_I"] = rel_frob(VO, I_head)
                values["cos_headspace_VO_I"] = cosine_to(VO, I_head)

                for m in metric_names:
                    layer_metric_scores[m].append(values[m])

                row = {
                    "layer": layer_idx,
                    "head": h,
                    "kv_head": kv_h,
                    "head_dim": head_dim,
                    "q_heads": q_heads,
                    "k_heads": k_heads,
                    "v_heads": v_heads,
                    "o_heads": o_heads,
                    "usable_heads": usable_heads,
                }
                row.update(values)
                rows.append(row)

            for m in metric_names:
                scores[m].append(layer_metric_scores[m])

    csv_path = os.path.join(args.outdir, "head_special_matrix_scores.csv")
    fieldnames = [
        "layer",
        "head",
        "kv_head",
        "head_dim",
        "q_heads",
        "k_heads",
        "v_heads",
        "o_heads",
        "usable_heads",
    ] + metric_names

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Saved CSV:", csv_path)

    descriptions = {
        "cos_identity_OV": "OV cosine to identity, higher = more identity-like",
        "rel_frob_identity_OV": "OV relative Frobenius distance to identity, lower = more identity-like",
        "frob_identity_OV": "OV Frobenius distance to identity, lower = more identity-like",

        "cos_negative_identity_OV": "OV cosine to negative identity, higher = more -I-like",
        "rel_frob_negative_identity_OV": "OV relative Frobenius distance to -I, lower = more -I-like",

        "rel_frob_orthogonal_OVtOV_I": "OVᵀOV relative distance to I, lower = more orthogonal/rotation-like",
        "cos_orthogonal_OVtOV_I": "OVᵀOV cosine to I, higher = more orthogonal/rotation-like",

        "rel_frob_involution_OV2_I": "OV² relative distance to I, lower = more inverse/involution-like",
        "cos_involution_OV2_I": "OV² cosine to I, higher = more inverse/involution-like",

        "rel_frob_projection_OV2_OV": "OV² relative distance to OV, lower = more projection-like",
        "cos_projection_OV2_OV": "OV² cosine to OV, higher = more projection-like",

        "rel_frob_symmetric_OV_OVt": "OV relative distance to OVᵀ, lower = more symmetric",
        "cos_symmetric_OV_OVt": "OV cosine to OVᵀ, higher = more symmetric",

        "rel_frob_skew_OV_negOVt": "OV relative distance to -OVᵀ, lower = more skew-symmetric",
        "cos_skew_OV_negOVt": "OV cosine to -OVᵀ, higher = more skew-symmetric",

        "rel_frob_headspace_VO_I": "V_hO_h relative distance to head identity, lower = more inverse-like in head space",
        "cos_headspace_VO_I": "V_hO_h cosine to head identity, higher = more inverse-like in head space",
    }

    for m in metric_names:
        arr = np.array(scores[m], dtype=float)

        path = os.path.join(args.outdir, f"{m}_heatmap.png")

        save_heatmap(
            arr,
            m,
            "Head",
            "Layer",
            descriptions[m],
            path,
            dpi=args.dpi,
            show=args.show,
        )

        print("Saved heatmap:", path)

    print("Done.")


if __name__ == "__main__":
    main()
