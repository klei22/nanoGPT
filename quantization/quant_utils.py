import torch
import torch.nn as nn

def set_variant(variant, default_variant):
    # If variant is false or None, then set to provided default value
    if not variant:
        return default_variant
    return variant

def create_activation_buffers(obj, arg):
    arg_str = arg.split("quantize_")[1]
    obj.register_buffer(arg_str, None)
    obj.register_buffer(f"{arg_str}_scale", None)
    obj.register_buffer(f"{arg_str}_zero_point", None)


def create_activation_clip_parameters(obj, arg, init_value):
    arg_str = arg.split("quantize_")[1]
    clip_min_name = f"{arg_str}_clip_min"
    clip_max_name = f"{arg_str}_clip_max"
    if not hasattr(obj, clip_min_name):
        obj.register_parameter(clip_min_name, nn.Parameter(torch.tensor(-init_value)))
    if not hasattr(obj, clip_max_name):
        obj.register_parameter(clip_max_name, nn.Parameter(torch.tensor(init_value)))


def compute_activation_kurtosis(tensor, eps=1e-6):
    mean = tensor.mean(dim=-1, keepdim=True)
    centered = tensor - mean
    var = centered.pow(2).mean(dim=-1)
    kurtosis = centered.pow(4).mean(dim=-1) / (var.pow(2) + eps)
    return kurtosis.mean()
