import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import datetime
import numpy as np
import pickle
from scipy.stats import kurtosis

def parse_args():
    parser = argparse.ArgumentParser(description='Load quantized weights and activations from ckpt')
    parser.add_argument("--file_name", type=str, default="quantized_weights.pkl", 
                        help="File name to load the quantized weights/activations, scale factor, and zero point from")
    parser.add_argument('--image_folder', type=str, default="images", help="Folder to place images")
    parser.add_argument('--weight', type=str, default="all", help="Which quantized weight or activation to display")
    parser.add_argument('--graph', type=str, choices=['histogram', 'matrix'], default='matrix',
                        help='Which graph to use: histogram or matrix')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.image_folder, exist_ok=True)

    extension = os.path.splitext(args.file_name)[1]
    quantized_dict = {}
    if extension == ".pkl":
        with open(args.file_name, 'rb') as f:
            quantized_dict = pickle.load(f)
            print("Loaded dictionary with keys: ", quantized_dict.keys())

    if args.weight == "all":
        to_display = quantized_dict.keys()
    else:
        to_display = [args.weight]

    for key in to_display:
        if ("weight_norm" not in key) and ("scale" not in key) and ("zero_point" not in key):
            arr = quantized_dict[key]
            sample = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
            # Plotting
            if args.graph == 'matrix':
                plt.figure(figsize=(10, 8))
                sns.heatmap(sample, cmap='viridis', annot=False)
                plt.title(f'{key} Matrix')
                plt.xlabel('Columns')
                plt.ylabel('Rows')
                image_path = os.path.join(args.image_folder, f'{key}_matrix_{timestamp}.png')
            elif args.graph == 'histogram':
                flatten = arr.flatten()
                mean = np.mean(np.isnan(flatten))
                std_dev = np.std(np.isnan(flatten))
                kurt = kurtosis(flatten, fisher=True)
                plt.hist(flatten, bins=50)
                plt.title(f'{key} Histogram\nMean: {mean:.4f}, Std Dev: {std_dev:.4f}, Kurtosis: {kurt:.4f}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                image_path = os.path.join(args.image_folder, f'{key}_histogram_{timestamp}.png')

            # Save the image
            plt.savefig(image_path)
            print(f'Saved image to {image_path}')

if __name__ == "__main__":
    main()