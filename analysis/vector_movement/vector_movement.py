import argparse
import numpy as np
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import plotly.graph_objects as go
from plotly.io import write_html
from pathlib import Path
import sys

# path setup (repo root) for potential imports if needed
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Utility from vector_distribution_analysis

def add_xyz_axes(fig, length=1.2, row=None, col=None):
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
        ), row=row, col=col)


def generate_mesh_from_healpix(nside):
    """Generate a triangle mesh from HEALPix pixel boundaries."""
    import healpy as hp
    npix = hp.nside2npix(nside)
    verts = []
    faces = []
    for pix in range(npix):
        corners = hp.boundaries(nside, pix, step=1).T
        base = len(verts)
        for vec in corners:
            verts.append(vec)
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    return np.array(verts), np.array(faces)


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


def vector_trace(vec, name, color, scene=None):
    trace = go.Scatter3d(
        x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
        mode='lines+markers',
        line=dict(color=color, width=6),
        marker=dict(size=4),
        name=name,
        showlegend=False
    )
    if scene is not None:
        trace.update(scene=scene)
    return trace


def make_animation(history, target, out_html):
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}, {'type': 'domain'}]],
        subplot_titles=(
            'Direct plot', 'HEALPix plot', 'XY', 'XZ', 'YZ', ''
        )
    )

    verts, faces = generate_mesh_from_healpix(8)
    fig.add_trace(
        go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                  i=faces[:,0], j=faces[:,1], k=faces[:,2],
                  opacity=0.2, color='lightgrey', showscale=False),
        row=1, col=2
    )

    add_xyz_axes(fig, row=1, col=1)
    add_xyz_axes(fig, row=1, col=2)

    target_tr1 = vector_trace(target, 'target', 'green', scene='scene1')
    target_tr2 = vector_trace(target, 'target', 'green', scene='scene2')

    frames = []
    for i, (raw_v, norm_v) in enumerate(history):
        data = [
            vector_trace(raw_v, 'vector', 'red', scene='scene1'),
            vector_trace(norm_v, 'norm', 'orange', scene='scene1'),
            target_tr1,
            vector_trace(raw_v, 'vector', 'red', scene='scene2'),
            vector_trace(norm_v, 'norm', 'orange', scene='scene2'),
            target_tr2,
            go.Scatter(x=[raw_v[0]], y=[raw_v[1]], mode='markers', marker=dict(color='red'), showlegend=False, xaxis='x3', yaxis='y3'),
            go.Scatter(x=[norm_v[0]], y=[norm_v[1]], mode='markers', marker=dict(color='orange'), showlegend=False, xaxis='x3', yaxis='y3'),
            go.Scatter(x=[target[0]], y=[target[1]], mode='markers', marker=dict(color='green'), showlegend=False, xaxis='x3', yaxis='y3'),
            go.Scatter(x=[raw_v[0]], y=[raw_v[2]], mode='markers', marker=dict(color='red'), showlegend=False, xaxis='x4', yaxis='y4'),
            go.Scatter(x=[norm_v[0]], y=[norm_v[2]], mode='markers', marker=dict(color='orange'), showlegend=False, xaxis='x4', yaxis='y4'),
            go.Scatter(x=[target[0]], y=[target[2]], mode='markers', marker=dict(color='green'), showlegend=False, xaxis='x4', yaxis='y4'),
            go.Scatter(x=[raw_v[1]], y=[raw_v[2]], mode='markers', marker=dict(color='red'), showlegend=False, xaxis='x5', yaxis='y5'),
            go.Scatter(x=[norm_v[1]], y=[norm_v[2]], mode='markers', marker=dict(color='orange'), showlegend=False, xaxis='x5', yaxis='y5'),
            go.Scatter(x=[target[1]], y=[target[2]], mode='markers', marker=dict(color='green'), showlegend=False, xaxis='x5', yaxis='y5')
        ]
        coord_text = (
            f"vector: {np.round(raw_v,3)}<br>"
            f"norm: {np.round(norm_v,3)}<br>"
            f"target: {np.round(target,3)}"
        )
        layout = go.Layout(annotations=[dict(x=0.83, y=0.25, xref='paper', yref='paper',
                                             text=coord_text, showarrow=False)])
        frames.append(go.Frame(data=data, layout=layout, name=str(i)))

    fig = go.Figure(frames=frames)
    fig.add_traces(frames[0].data)
    if frames[0].layout.annotations:
        fig.update_layout(annotations=frames[0].layout.annotations)

    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]}]
        }],
        scene1=dict(camera=dict(projection=dict(type='perspective')), aspectmode='cube',
                    xaxis=dict(range=[-1.5,1.5]), yaxis=dict(range=[-1.5,1.5]), zaxis=dict(range=[-1.5,1.5])),
        scene2=dict(camera=dict(projection=dict(type='perspective')), aspectmode='cube',
                    xaxis=dict(range=[-1.5,1.5]), yaxis=dict(range=[-1.5,1.5]), zaxis=dict(range=[-1.5,1.5])),
        xaxis3=dict(range=[-1.5,1.5]),
        yaxis3=dict(range=[-1.5,1.5]),
        xaxis4=dict(range=[-1.5,1.5]),
        yaxis4=dict(range=[-1.5,1.5]),
        xaxis5=dict(range=[-1.5,1.5]),
        yaxis5=dict(range=[-1.5,1.5])
    )

    fig.frames = frames
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
    p.add_argument('--init-x', type=float)
    p.add_argument('--init-y', type=float)
    p.add_argument('--init-z', type=float)
    p.add_argument('--target-x', type=float)
    p.add_argument('--target-y', type=float)
    p.add_argument('--target-z', type=float)
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

    if args.init_x is not None and args.init_y is not None and args.init_z is not None:
        init_v = np.array([args.init_x, args.init_y, args.init_z], dtype=np.float32)
    elif args.init_preset:
        init_v = preset_vector(args.init_preset)
    else:
        init_v = np.random.normal(loc=args.init_mean, scale=args.init_std, size=3).astype(np.float32)

    if args.target_x is not None and args.target_y is not None and args.target_z is not None:
        target_v = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)
    elif args.target_preset:
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
