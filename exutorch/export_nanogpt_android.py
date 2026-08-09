"""
export_nanogpt_android.py

Converts nanoGPT (GPT-2 weights or a local checkpoint) into two ExecuTorch
.pte bundles ready to copy onto an Android device:

  nanogpt_fp32.pte    – FP32 model delegated to XNNPack (ARM NEON)
  nanogpt_int8.pte    – INT8-weight-quantised model delegated to XNNPack

Both files can be loaded by the companion Android app with the same Java API;
the int8 variant is ~4× smaller and faster on most Arm Cortex-A cores.

Usage
-----
# GPT-2 124 M (default)
python export_nanogpt_android.py

# Larger variant
python export_nanogpt_android.py --model_type gpt2-medium

# Local checkpoint from train.py
python export_nanogpt_android.py --checkpoint ../out/ckpt.pt

# Skip the INT8 export (faster iteration)
python export_nanogpt_android.py --no_int8

# Output directory
python export_nanogpt_android.py --out_dir /tmp/android_models

Design notes
------------
* We export the full-context (sliding-window) version of the model.
  The Android app measures TTFT as the first forward pass, and decode
  latency as subsequent forward passes, which is correct even without
  an explicit KV-cache.
* Dynamic shapes are set so the sequence dimension can be anything from
  1 up to block_size-1, which lets the Android runner feed growing
  context windows.
* XNNPack partitioner is applied after quantisation so INT8 matmuls are
  lowered to XNNPack's optimised Arm NEON kernels.
"""

import argparse
import os
import sys

import torch
from torch.export import export, export_for_training
from torch.nn.attention import SDPBackend, sdpa_kernel

# ExecuTorch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config

# PT2E INT8 weight-only quantisation
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_from_pretrained(model_type: str):
    """Load GPT-2 weights from HuggingFace via the exutorch/model.py shim."""
    # exutorch/model.py is a self-contained copy of nanoGPT, no extra deps.
    sys.path.insert(0, os.path.dirname(__file__))
    from model import GPT  # exutorch/model.py
    model = GPT.from_pretrained(model_type)
    model.eval()
    return model


def load_from_checkpoint(ckpt_path: str):
    """Load a checkpoint saved by train.py (uses the full model.py)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    from model import GPT
    from gpt_conf import GPTConfig

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gptconf = GPTConfig(**ckpt["model_args"])
    model   = GPT(gptconf)

    # Strip DDP / compile prefixes that train.py may have added
    raw_sd = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(raw_sd, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _make_example_inputs(model) -> tuple:
    block_size = model.config.block_size
    return (torch.randint(0, 100, (1, block_size - 1), dtype=torch.long),)


def _make_dynamic_shapes(model) -> tuple:
    block_size = model.config.block_size
    return (
        {1: torch.export.Dim("token_dim", min=1, max=block_size - 1)},
    )


def export_fp32(model, out_path: str) -> None:
    """Export a plain FP32 model with XNNPack delegation."""
    print("\n[FP32] Tracing model with torch.export …")
    example_inputs  = _make_example_inputs(model)
    dynamic_shapes  = _make_dynamic_shapes(model)

    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m = export_for_training(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        ).module()
        traced = export(m, example_inputs, dynamic_shapes=dynamic_shapes)

    print("[FP32] Lowering to Edge IR …")
    edge_cfg = get_xnnpack_edge_compile_config()
    edge_mgr = to_edge(traced, compile_config=edge_cfg)

    print("[FP32] Delegating to XNNPack …")
    edge_mgr = edge_mgr.to_backend(XnnpackPartitioner())
    et_prog  = edge_mgr.to_executorch()

    with open(out_path, "wb") as f:
        f.write(et_prog.buffer)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[FP32] Saved → {out_path}  ({size_mb:.1f} MB)")


def export_int8(model, out_path: str) -> None:
    """
    Export an INT8 weight-only quantised model with XNNPack delegation.

    We use PT2E (Pytorch 2 Export) quantisation:
      1. prepare_pt2e:  insert observer / fake-quant nodes
      2. calibrate:     one forward pass with example data (no real data needed
                        for weight-only quant)
      3. convert_pt2e:  fold observers into quantised ops
      4. to_edge / to_backend / to_executorch as usual
    """
    print("\n[INT8] Setting up XNNPACKQuantizer (weight-only, per-channel) …")
    example_inputs = _make_example_inputs(model)
    dynamic_shapes = _make_dynamic_shapes(model)

    # Clone so we don't mutate the original model for the FP32 path
    model_q = type(model)(model.config)
    model_q.load_state_dict(model.state_dict())
    model_q.eval()

    # Trace to get the ATen graph for quantisation
    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m = export_for_training(
            model_q, example_inputs, dynamic_shapes=dynamic_shapes
        ).module()

    # Weight-only INT8 symmetric quantisation (no activation quant)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,   # static / weight-only
            is_qat=False,
        )
    )

    print("[INT8] Inserting quantisation observers …")
    m_prepared = prepare_pt2e(m, quantizer)

    # Calibration – one forward pass is sufficient for weight-only quant
    with torch.no_grad():
        m_prepared(*example_inputs)

    print("[INT8] Converting to quantised model …")
    m_quantised = convert_pt2e(m_prepared)

    print("[INT8] Re-tracing quantised model …")
    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        traced = export(m_quantised, example_inputs, dynamic_shapes=dynamic_shapes)

    print("[INT8] Lowering to Edge IR …")
    edge_cfg = get_xnnpack_edge_compile_config()
    edge_mgr = to_edge(traced, compile_config=edge_cfg)

    print("[INT8] Delegating to XNNPack …")
    edge_mgr = edge_mgr.to_backend(XnnpackPartitioner())
    et_prog  = edge_mgr.to_executorch()

    with open(out_path, "wb") as f:
        f.write(et_prog.buffer)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[INT8] Saved → {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# KV-cache export
# ---------------------------------------------------------------------------

def export_kvcache(model_type: str, out_dir: str) -> None:
    """
    Export a KV-cache enabled model for efficient on-device autoregressive
    generation.  Produces two .pte files:

      nanogpt_kvcache_prefill.pte   – processes the full prompt
      nanogpt_kvcache_decode.pte    – processes one token at a time

    The KV-cache state persists inside the .pte module buffers between
    successive decode calls, so no external state management is needed.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from nanogpt_kvcache import GPTWithKVCache

    print("\n[KVCache] Loading model …")
    kv_model = GPTWithKVCache.from_pretrained(model_type)
    kv_model.eval()

    block_size = kv_model.config.block_size

    # ---- Prefill export ---------------------------------------------------
    print("[KVCache] Exporting prefill model …")
    # Example: process a prompt of 32 tokens starting at position 0
    prefill_tokens    = torch.randint(0, 100, (1, 32), dtype=torch.long)
    prefill_start_pos = 0
    prefill_is_causal = True

    prefill_inputs    = (prefill_tokens, prefill_start_pos, prefill_is_causal)
    prefill_dyn       = ({1: torch.export.Dim("prompt_len", min=1, max=block_size - 1)}, None, None)

    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m_pre = export_for_training(
            kv_model, prefill_inputs, dynamic_shapes=prefill_dyn
        ).module()
        traced_prefill = export(m_pre, prefill_inputs, dynamic_shapes=prefill_dyn)

    edge_cfg = get_xnnpack_edge_compile_config()
    edge_pre = to_edge(traced_prefill, compile_config=edge_cfg).to_backend(XnnpackPartitioner())
    et_pre   = edge_pre.to_executorch()

    prefill_path = os.path.join(out_dir, "nanogpt_kvcache_prefill.pte")
    with open(prefill_path, "wb") as f:
        f.write(et_pre.buffer)
    print(f"[KVCache] Prefill saved → {prefill_path}")

    # ---- Decode export ----------------------------------------------------
    print("[KVCache] Exporting decode model …")
    # Reload a fresh model (KV cache is reset) for the decode trace
    kv_model_dec = GPTWithKVCache.from_pretrained(model_type)
    kv_model_dec.eval()

    decode_token     = torch.randint(0, 100, (1, 1), dtype=torch.long)
    decode_start_pos = 32  # position after the 32-token prefill
    decode_is_causal = False   # single-token decode → no causal mask needed

    decode_inputs = (decode_token, decode_start_pos, decode_is_causal)
    # No dynamic shapes for decode – always exactly 1 token
    decode_dyn    = (None, None, None)

    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m_dec = export_for_training(
            kv_model_dec, decode_inputs, dynamic_shapes=decode_dyn
        ).module()
        traced_decode = export(m_dec, decode_inputs, dynamic_shapes=decode_dyn)

    edge_dec = to_edge(traced_decode, compile_config=edge_cfg).to_backend(XnnpackPartitioner())
    et_dec   = edge_dec.to_executorch()

    decode_path = os.path.join(out_dir, "nanogpt_kvcache_decode.pte")
    with open(decode_path, "wb") as f:
        f.write(et_dec.buffer)
    print(f"[KVCache] Decode saved → {decode_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export nanoGPT to ExecuTorch .pte for Android",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_type",
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="HuggingFace GPT-2 variant (ignored if --checkpoint is set)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a local train.py checkpoint (overrides --model_type)",
    )
    p.add_argument(
        "--out_dir",
        default=".",
        help="Directory for the output .pte files",
    )
    p.add_argument(
        "--no_fp32",
        action="store_true",
        help="Skip the FP32 export",
    )
    p.add_argument(
        "--no_int8",
        action="store_true",
        help="Skip the INT8 quantised export",
    )
    p.add_argument(
        "--kvcache",
        action="store_true",
        help="Also export KV-cache prefill+decode models (XNNPack, FP32 only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load model -------------------------------------------------------
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_from_checkpoint(args.checkpoint)
    else:
        print(f"Loading pretrained: {args.model_type}")
        model = load_from_pretrained(args.model_type)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f} M")
    print(f"Block size:  {model.config.block_size}")
    print(f"Vocab size:  {model.config.vocab_size}")
    print(f"Layers:      {model.config.n_layer}  "
          f"Heads: {model.config.n_head}  "
          f"Embd: {model.config.n_embd}")

    # ---- FP32 export ------------------------------------------------------
    if not args.no_fp32:
        export_fp32(model, os.path.join(args.out_dir, "nanogpt_fp32.pte"))

    # ---- INT8 export ------------------------------------------------------
    if not args.no_int8:
        export_int8(model, os.path.join(args.out_dir, "nanogpt_int8.pte"))

    # ---- KV-cache exports -------------------------------------------------
    if args.kvcache:
        if args.checkpoint:
            print("[KVCache] WARNING: KV-cache export uses nanogpt_kvcache.py "
                  "which only supports GPT-2 HuggingFace weights.  "
                  "Skipping KV-cache export for checkpoint models.")
        else:
            export_kvcache(args.model_type, args.out_dir)

    print("\nDone.  Files written to:", os.path.abspath(args.out_dir))
    for fname in sorted(os.listdir(args.out_dir)):
        if fname.endswith(".pte"):
            full = os.path.join(args.out_dir, fname)
            print(f"  {fname:45s}  {os.path.getsize(full)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
