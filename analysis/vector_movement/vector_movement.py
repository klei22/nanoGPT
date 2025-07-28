import argparse
import numpy as np
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import plotly.graph_objects as go
from plotly.io import write_html


# Utility from vector_distribution_analysis

def add_xyz_axes(fig, length=1.2):
    axes = {
        'x': ([0, length], [0, 0], [0, 0], 'red'),
        'y': ([0, 0], [0, length], [0, 0], 'green'),
        'z': ([0, 0], [0, 0], [0, length], 'blue'),
    }
    for name, (x, y, z, color) in axes.items():
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+text',
            line=dict(color=color, width=4),
            text=[None, name],
            textposition='top center',
            showlegend=False
        ))


def fake_quantize_int(v: torch.Tensor, bits: int) -> torch.Tensor:
    """Round tensor to nearest integer within the given bit width."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    return torch.round(v).clamp(qmin, qmax)


def rms_norm_to_unit(vec, eps=1e-8):
    dim = vec.numel()
    rms = torch.sqrt(torch.mean(vec ** 2) + eps)
    gain = 1.0 / np.sqrt(dim)
    return vec * (gain / rms)


def vector_trace(vec, name, color):
    return go.Scatter3d(
        x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
        mode='lines+markers',
        line=dict(color=color, width=6),
        marker=dict(size=4),
        name=name
    )


def make_animation(history, target, out_html):
    target_trace = vector_trace(target, 'target', 'green')
    frames = []
    for i, (raw_v, norm_v) in enumerate(history):
        frame = go.Frame(
            data=[
                vector_trace(raw_v, 'vector', 'red'),
                vector_trace(norm_v, 'norm', 'orange'),
                target_trace
            ],
            name=str(i)
        )
        frames.append(frame)

    fig = go.Figure(frames=frames)
    fig.add_traces(frames[0].data)
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]}]
        }],
        scene=dict(aspectmode='cube', xaxis=dict(range=[-1.5,1.5]), yaxis=dict(range=[-1.5,1.5]), zaxis=dict(range=[-1.5,1.5]))
    )
    add_xyz_axes(fig)
    write_html(fig, out_html)
    print(f"Saved animation to {out_html}")


def preset_vector(name):
    if name == 'x':
        return np.array([1., 0., 0.], dtype=np.float32)
    if name == 'y':
        return np.array([0., 1., 0.], dtype=np.float32)
    if name == 'z':
        return np.array([0., 0., 1.], dtype=np.float32)
    if name == 'corner':
        return np.array([1., 1., 1.], dtype=np.float32) / np.sqrt(3.0)
    raise ValueError(f"unknown preset {name}")


def parse_args():
    p = argparse.ArgumentParser(description='Vector movement toward target')
    p.add_argument('--init-mean', type=float, default=0.0)
    p.add_argument('--init-std', type=float, default=1.0)
    p.add_argument('--target-mean', type=float, default=0.0)
    p.add_argument('--target-std', type=float, default=1.0)
    p.add_argument('--init-preset', choices=['x','y','z','corner'])
    p.add_argument('--target-preset', choices=['x','y','z','corner'])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--lr-schedule', choices=['constant','cosine'], default='constant')
    p.add_argument('--optimizer', choices=['adamw','adam','sgd'], default='adamw')
    p.add_argument('--loss', choices=['dot','cosine'], default='dot')
    p.add_argument('--no-norm', action='store_true', help='disable RMS normalization')
    p.add_argument('--quant-bits', type=int, default=4, choices=range(3, 9),
                   metavar='B', help='fake quantization bit width (3-8)')
    p.add_argument('--out', default='vector_movement.html')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    if args.init_preset:
        init_v = preset_vector(args.init_preset)
    else:
        init_v = np.random.normal(loc=args.init_mean, scale=args.init_std, size=3).astype(np.float32)
    if args.target_preset:
        target_v = preset_vector(args.target_preset)
    else:
        target_v = np.random.normal(loc=args.target_mean, scale=args.target_std, size=3).astype(np.float32)

    v = torch.tensor(init_v, requires_grad=True)
    v.data = fake_quantize_int(v.data, args.quant_bits)
    target = torch.tensor(target_v)
    target = fake_quantize_int(target, args.quant_bits)
    target_unit = target / (target.norm() + 1e-8)

    if args.optimizer == 'adamw':
        opt = AdamW([v], lr=args.lr)
    elif args.optimizer == 'adam':
        opt = Adam([v], lr=args.lr)
    else:
        opt = SGD([v], lr=args.lr)

    if args.lr_schedule == 'cosine':
        lr_lambda = lambda step: 0.5 * (1 + np.cos(np.pi * step / args.steps))
        sched = LambdaLR(opt, lr_lambda)
    else:
        sched = None

    history = []
    for step in range(args.steps):
        opt.zero_grad()
        vec_for_loss = v
        if not args.no_norm:
            vec_for_loss = rms_norm_to_unit(v)
        if args.loss == 'dot':
            loss = -torch.dot(vec_for_loss, target_unit)
        else:
            loss = 1 - torch.nn.functional.cosine_similarity(vec_for_loss.unsqueeze(0), target_unit.unsqueeze(0))
            loss = loss.squeeze()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v.data = fake_quantize_int(v.data, args.quant_bits)
        if sched:
            sched.step()
        with torch.no_grad():
            norm_v = rms_norm_to_unit(v) if not args.no_norm else v.clone()
            history.append((v.detach().clone().numpy(), norm_v.detach().clone().numpy()))

    make_animation(history, target_unit.numpy(), args.out)


if __name__ == '__main__':
    main()
