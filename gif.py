import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

def visualize_frame(players, ball, player_size, ball_size, ball_scale_with_z):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

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

    plt.title(f"Soccer Field Visualization (Ball Z: {bz:.2f})")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize soccer player and ball positions.")
    parser.add_argument("csv_file", help="Path to the CSV file containing position data.")
    parser.add_argument("--player_size", type=float, default=2.0, help="Size of player circles.")
    parser.add_argument("--ball_size", type=float, default=1.0, help="Base size of the ball circle.")
    parser.add_argument("--ball_scale", action="store_true", help="Scale ball size with z-coordinate.")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to visualize (0-indexed).")
    args = parser.parse_args()

    data = read_csv(args.csv_file)
    if args.frame >= len(data):
        print(f"Error: Frame {args.frame} does not exist. Max frame is {len(data) - 1}.")
        return

    row = data[args.frame]
    players = parse_player_data(row)
    ball = parse_ball_data(row)

    visualize_frame(players, ball, args.player_size, args.ball_size, args.ball_scale)

if __name__ == "__main__":
    main()
