import torch
import torch.nn as nn
import argparse

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import GPT
from gpt_conf import GPTConfig
from variations.activation_variations import GELUShifted  # Import GELUShifted


def load_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)

    state_dict = checkpoint['model']
    for k, v in list(state_dict.items()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model

def find_gelushifted(model):
    gelu_shifted_instances = []

    def recurse_modules(module, prefix=''):
        for name, child in module.named_children():
            child_prefix = f'{prefix}.{name}' if prefix else name
            if isinstance(child, GELUShifted):
                gelu_shifted_instances.append((child_prefix, child))
            else:
                recurse_modules(child, child_prefix)
    recurse_modules(model)
    return gelu_shifted_instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find GELUShifted modules and print their shift parameter.")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the model checkpoint.")
    args = parser.parse_args()

    model = load_checkpoint(args.ckpt_path)

    gelu_shifted_modules = find_gelushifted(model)

    if not gelu_shifted_modules:
        print("No GELUShifted modules found in the model.")
    else:
        for name, module in gelu_shifted_modules:
            print(f"\nFound GELUShifted module at: {name}")
            if hasattr(module, 'shift'):
                shift_value = module.shift
                if isinstance(shift_value, nn.Parameter):
                    shift_value = shift_value.detach().cpu().numpy()
                else:
                    shift_value = shift_value.data.cpu().numpy()
                print(f"Shift parameter value: {shift_value}")
            else:
                print("No shift parameter found in the module.")

