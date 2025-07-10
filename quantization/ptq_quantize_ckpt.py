import argparse
import os
import torch

from model import GPT, GPTConfig

from quantization.quantize import quantize_dictionary


def parse_args():
    p = argparse.ArgumentParser(description="Post-training quantize a checkpoint")
    p.add_argument("--ckpt_path", type=str, default=os.path.join("out", "ckpt.pt"),
                   help="Path to checkpoint produced by train.py")
    p.add_argument("--out_ckpt", type=str, default=os.path.join("out", "ckpt_ptq.pt"),
                   help="Where to save the quantized checkpoint")
    p.add_argument("--bits", type=int, default=4, help="Quantization bit width")
    p.add_argument("--quant_method", type=str, default="symmetric_quant",
                   choices=list(quantize_dictionary.keys()),
                   help="Quantization method for linear weights")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device used for quantization computations")
    return p.parse_args()


def load_model(ckpt, device):
    model_args = ckpt["model_args"]
    model_args["dropout"] = 0.0
    model_args["linear_variant_attn"] = "quantized_linear"
    model_args["linear_variant_mlp"] = "quantized_linear"
    # configure quantization
    model_args["quantize_linear_bits"] = args.bits
    model_args["quantize_linear_method"] = args.quant_method
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state.keys()):
        if k.startswith(unwanted_prefix):
            state[k[len(unwanted_prefix):]] = state.pop(k)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def quantize_model(model):
    for module in model.modules():
        if hasattr(module, "_eval"):
            module._eval()


def main(args):
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model = load_model(ckpt, args.device)
    quantize_model(model)
    ckpt["model"] = model.state_dict()
    torch.save(ckpt, args.out_ckpt)
    print(f"Quantized checkpoint saved to {args.out_ckpt}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

