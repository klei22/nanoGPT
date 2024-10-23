
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model import GPT  # Assuming the GPT model is in a file named 'gpt_model.py'
from gpt_conf import GPTConfig
from variations.activation_variations import *
from scipy.interpolate import CubicSpline

def load_checkpoint(ckpt_path):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Load model config and initialize model
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)  # Modify this if needed to fit your config
    model = GPT(config)

    # Load model state dict
    state_dict = checkpoint['model']

    # Update state_dict keys if they start with '_orig_mod.' prefix
    for k, v in list(state_dict.items()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

    # Load model state
    model.load_state_dict(state_dict)

    return model

import pywt  # Import PyWavelets

def plot_with_and_without_gelu(x_vals, y_vals, i, output_dir):
    """
    Generate two plots: one with the GeLU overlay and one without.
    Also, generate a continuous wavelet spectrogram of the difference.
    """
    x_nlim = -25
    x_lim = 25
    y_nlim = -.25
    y_lim = 25

    # Create a 1000-point array for the GeLU graph overlay for smoothness
    x_gelu_smooth = np.linspace(x_nlim, x_lim, 1000)
    gelu_fn = torch.nn.GELU()
    gelu_y_vals_smooth = gelu_fn(torch.tensor(x_gelu_smooth)).detach().cpu().numpy()

    # Plot without GeLU overlay
    plt.figure()
    plt.scatter(x_vals, y_vals, color='blue', label='Piecewise Points')
    plt.plot(x_vals, y_vals, linestyle='--', color='cyan', label='Linear Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Piecewise Activation for Layer {i} (No GeLU Overlay)')
    plt.legend()
    plot_path_no_gelu = os.path.join(output_dir, f'layer_{i}_piecewise_activation_no_gelu.png')
    plt.xlim(x_nlim, x_lim)
    plt.ylim(y_nlim, y_lim)
    plt.savefig(plot_path_no_gelu)
    plt.close()
    print(f"Saved plot for Layer {i} without GeLU overlay to {plot_path_no_gelu}")

    # Compute the difference between the learned function and GeLU
    from scipy.interpolate import CubicSpline

    # Convert tensors to numpy for scipy CubicSpline
    x_np = x_vals
    y_np = y_vals

    # Create cubic spline
    spline = CubicSpline(x_np, y_np)

    # Interpolate values smoothly between points
    x_new = np.linspace(min(x_np), max(x_np), 1000)  # More points for smooth interpolation
    y_new = spline(x_new)
    y_diff = y_new - gelu_fn(torch.tensor(x_new)).detach().cpu().numpy()

    # Plot the difference (for visualization purposes)
    plt.figure()
    plt.plot(x_new, y_diff, '-', label='Diff of GELU and Learned Function')
    plt.legend()
    plot_diff_path = os.path.join(output_dir, f'layer_{i}_difference_plot.png')
    plt.savefig(plot_diff_path)
    plt.close()
    print(f"Saved difference plot for Layer {i} to {plot_diff_path}")

    # Compute Continuous Wavelet Transform (CWT) for the difference
    # wavelet = 'cmor'  # Complex Morlet wavelet
    wavelet = 'mexh'  # Complex Morlet wavelet
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(y_diff, scales, wavelet)

    # Plot the wavelet spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[x_new[0], x_new[-1], frequencies[-1], frequencies[0]],
               cmap='jet', aspect='auto', vmax=abs(coefficients).max())
    plt.colorbar(label='Magnitude')
    plt.xlabel('x')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Wavelet Spectrogram of the Difference for Layer {i}')
    plot_wavelet_path = os.path.join(output_dir, f'layer_{i}_wavelet_spectrogram.png')
    plt.savefig(plot_wavelet_path)
    plt.close()
    print(f"Saved wavelet spectrogram for Layer {i} to {plot_wavelet_path}")

    # Plot with GeLU overlay (black for GeLU)
    plt.figure()
    plt.scatter(x_vals, y_vals, color='blue', label='Piecewise Points')
    plt.plot(x_vals, y_vals, linestyle='--', color='cyan', label='Linear Interpolation')
    plt.plot(x_gelu_smooth, gelu_y_vals_smooth, color='black', label='GeLU Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Piecewise Activation with GeLU Overlay for Layer {i}')
    plt.legend()
    plot_path_with_gelu = os.path.join(output_dir, f'layer_{i}_piecewise_activation_with_gelu.png')
    plt.xlim(x_nlim, x_lim)
    plt.ylim(y_nlim, y_lim)
    plt.savefig(plot_path_with_gelu)
    plt.close()
    print(f"Saved plot for Layer {i} with GeLU overlay to {plot_path_with_gelu}")


def print_piecewise_activation_params(model, output_dir='output_plots'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse the model layers and find the piecewise activation function
    for i, block in enumerate(model.transformer.h):
        if hasattr(block.mlp, 'activation_variant') and isinstance(block.mlp.activation_variant, PiecewiseFullyLearnableActivation):
            activation_fn = block.mlp.activation_variant
            x_vals = activation_fn.x_vals.detach().cpu().numpy()
            y_vals = activation_fn.y_vals.detach().cpu().numpy()

            # Extend to -5 and +5, assuming 0 for x <= -2 and y = x for x >= 2
            # x_vals = np.concatenate(([-5, -2], x_vals, [2, 5]))
            # y_vals = np.concatenate(([0, 0], y_vals, [2, 5]))

            # Create both versions (with and without GeLU overlay)
            plot_with_and_without_gelu(x_vals, y_vals, i, output_dir)
        break

if __name__ == "__main__":
    # Path to the checkpoint file
    # ckpt_path = './14000_399pfla.pt'
    ckpt_path = './16000_399pfla.pt'
    # ckpt_path = './9000.pt'

    # Load the model
    model = load_checkpoint(ckpt_path)

    # Print the piecewise activation parameters and save plots
    print_piecewise_activation_params(model)

