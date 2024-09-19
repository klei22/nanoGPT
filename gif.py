import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import imageio
import io

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def parse_player_data(row):
    players = {'home': [], 'away': []}
    for item in row[5:]:
        if item.startswith('H'):
            parts = item.split('X')
            number = int(parts[0][1:])
            x = float(parts[1].split('Y')[0])
            y = float(parts[1].split('Y')[1])
            players['home'].append((number, x, y))
        elif item.startswith('A'):
            parts = item.split('X')
            number = int(parts[0][1:])
            x = float(parts[1].split('Y')[0])
            y = float(parts[1].split('Y')[1])
            players['away'].append((number, x, y))
    return players

def parse_ball_data(row):
    bx = float(row[-3][2:])
    by = float(row[-2][2:])
    bz = float(row[-1][2:])
    return bx, by, bz

import matplotlib.image as mpimg

def create_frame(players, ball, player_size, ball_size, ball_scale_with_z):
    # Load the background image
    img = mpimg.imread('court.png')

    fig, ax = plt.subplots(figsize=(9.39, 5.0))  # Set figure size according to the desired aspect ratio
    ax.set_xlim(0, 100)  # X-axis remains from 0 to 100
    ax.set_ylim(0, 53.24)  # Adjust Y-axis to fit the aspect ratio
    ax.set_aspect(1.878)  # Set aspect ratio to match the 939x500 ratio

    # Display the background image
    ax.imshow(img, extent=[0, 100, 0, 53.24], aspect='auto')

    # Plot home players
    for number, x, y in players['home']:
        circle = Circle((x, y), player_size, facecolor='blue', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, str(number), ha='center', va='center', color='white')

    # Plot away players
    for number, x, y in players['away']:
        circle = Circle((x, y), player_size, facecolor='red', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, str(number), ha='center', va='center', color='white')

    # Plot ball
    bx, by, bz = ball
    if ball_scale_with_z:
        ball_radius = ball_size * (bz / 100 + 0.5)  # Scale between 0.5x and 1.5x of ball_size
    else:
        ball_radius = ball_size
    ball_circle = Circle((bx, by), ball_radius, facecolor='orange', edgecolor='black')
    ax.add_patch(ball_circle)

    plt.title(f"Basketball Forecast Visualization (Ball Z: {bz:.2f})")

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = imageio.imread(buf)
    buf.close()
    plt.close(fig)

    return image

def create_gif(data, output_file, player_size, ball_size, ball_scale_with_z, fps):
    images = []
    for i, row in enumerate(data):
        players = parse_player_data(row)
        ball = parse_ball_data(row)
        image = create_frame(players, ball, player_size, ball_size, ball_scale_with_z)
        images.append(image)
        print(f"Processed frame {i+1}/{len(data)}")

    imageio.mimsave(output_file, images, fps=fps)
    print(f"GIF saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize soccer player and ball positions and create a GIF.")
    parser.add_argument("csv_file", help="Path to the CSV file containing position data.")
    parser.add_argument("--player_size", type=float, default=2.0, help="Size of player circles.")
    parser.add_argument("--ball_size", type=float, default=1.0, help="Base size of the ball circle.")
    parser.add_argument("--ball_scale", action="store_true", help="Scale ball size with z-coordinate.")
    parser.add_argument("--output", default="soccer_animation.gif", help="Output GIF file name.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF.")
    args = parser.parse_args()

    data = read_csv(args.csv_file)
    create_gif(data, args.output, args.player_size, args.ball_size, args.ball_scale, args.fps)

if __name__ == "__main__":
    main()
