# utils/model_stats.py
import torch
from scipy.stats import kurtosis


def compute_weight_stats(model):
    stats = {}
    all_vals = []
    for name, param in model.named_parameters():
        data = param.detach().float().cpu()
        all_vals.append(data.view(-1))
        flat = data.view(-1).numpy()
        stats[name] = {
            'stdev': torch.std(data).item(),
            'kurtosis': float(kurtosis(flat, fisher=False)),
            'max': torch.max(data).item(),
            'min': torch.min(data).item(),
        }
    overall = {
        'stdev': 0.0,
        'kurtosis': 0.0,
        'max': 0.0,
        'min': 0.0,
    }
    if all_vals:
        cat = torch.cat(all_vals)
        cat_np = cat.numpy()
        overall = {
            'stdev': torch.std(cat).item(),
            'kurtosis': float(kurtosis(cat_np, fisher=False)),
            'max': torch.max(cat).item(),
            'min': torch.min(cat).item(),
        }
    return stats, overall


def compute_activation_stats(model, inputs, targets=None, iter_num=None):
    activation_stats = {}
    all_tensors = []
    hooks = []

    def register_hook(module, name):
        def hook_fn(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            data = tensor.detach().float().cpu()
            all_tensors.append(data.view(-1))
            flat = data.view(-1).numpy()
            activation_stats[name] = {
                'stdev': torch.std(data).item(),
                'kurtosis': float(kurtosis(flat, fisher=False)),
                'max': torch.max(data).item(),
                'min': torch.min(data).item(),
            }
        hooks.append(module.register_forward_hook(hook_fn))

    for name, module in model.named_modules():
        if len(list(module.parameters(recurse=False))) > 0:
            register_hook(module, name)

    model.eval()
    with torch.no_grad():
        if targets is not None:
            model(inputs, targets=targets, iter_num=iter_num)
        else:
            model(inputs, iter_num=iter_num)
    model.train()

    for h in hooks:
        h.remove()

    overall = {'stdev': 0.0, 'kurtosis': 0.0, 'max': 0.0, 'min': 0.0}
    if all_tensors:
        cat = torch.cat(all_tensors)
        cat_np = cat.numpy()
        overall = {
            'stdev': torch.std(cat).item(),
            'kurtosis': float(kurtosis(cat_np, fisher=False)),
            'max': torch.max(cat).item(),
            'min': torch.min(cat).item(),
        }

    return activation_stats, overall

