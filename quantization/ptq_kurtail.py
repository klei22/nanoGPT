import argparse
import os
import torch
from quantization.quantize import kurtail_quantize


def parse_args():
    p = argparse.ArgumentParser(description="Post-training quantization with KurTail")
    p.add_argument("--ckpt", type=str, default="out/ckpt.pt", help="Path to checkpoint")
    p.add_argument("--output", type=str, default="ptq_kurtail.pt", help="Output checkpoint path")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"]
    for name, tensor in list(state.items()):
        if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16) and tensor.ndim >= 2:
            zp, scale, q = kurtail_quantize(tensor.float(), args.bits)
            state[name] = q
            state[name + "_scale"] = scale
            state[name + "_zero_point"] = zp
    torch.save(ckpt, args.output)
    print(f"Saved quantized checkpoint to {args.output}")


if __name__ == "__main__":
    main()

