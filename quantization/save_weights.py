import torch
import argparse
import os
import datetime
import numpy as np
import pickle
import ml_dtypes
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='Load quantized weights and activations from ckpt')
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to load checkpoint from")
    parser.add_argument("--file_name", type=str, default="quantized_weights", help="File name to export the quantized weights/activations, scale factor, and zero point")
    parser.add_argument('--file_type', type=str, default="pkl", choices=["pkl"], help='Choose the file type to save quantized values')
    return parser.parse_args()

def save_quantized_data(state_dict, out_file, file_type):
    to_save = OrderedDict()
    for k, v in list(state_dict.items()):
        if "mlp_act" in k or "attn_act" in k or k.endswith("quantized_bias") or k.endswith("bias_norm") or k.endswith("zero_point") or k.endswith("quantized_weight") or k.endswith("weight_norm"):
            if v.dtype == torch.bfloat16:
                to_save[k] = v.cpu().float().numpy().astype(ml_dtypes.bfloat16)
            else:
                to_save[k] = v.cpu().numpy()

    if file_type == "pkl":
        with open(f"{out_file}.pkl", 'wb') as f:
            pickle.dump(to_save, f)

def main():
    args = parse_args()
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    save_quantized_data(state_dict, args.file_name, args.file_type)

if __name__ == "__main__":
    main()