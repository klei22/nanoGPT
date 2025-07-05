import numpy as np
import argparse

def generate_periodic_functions(points_per_cycle=100, num_cycles=10, num_functions=4):
    """Generate different periodic functions with varying frequencies and phases.
    
    Args:
        points_per_cycle (int): Number of points to sample per cycle
        num_cycles (int): Number of cycles to generate for each function
        num_functions (int): Number of different functions to generate
    
    Returns:
        np.ndarray: Array of shape (total_points, num_functions) containing the generated functions
    """
    total_points = points_per_cycle * num_cycles
    x = np.linspace(0, 2*np.pi, total_points)  # Scale x for one cycle
    functions = []

    # Basic sine wave (1 cycle per 2π)
    functions.append(np.sin(x * num_cycles))

    # Cosine wave with double frequency
    functions.append(np.cos(2 * x * num_cycles))

    # Composite wave (sum of sines)
    functions.append(0.5 * np.sin(x * num_cycles) + 0.3 * np.sin(3 * x * num_cycles))

    # Square wave approximation using Fourier series
    square = np.zeros_like(x)
    for k in range(1, 10, 2):
        square += (4/np.pi) * (np.sin(k * x * num_cycles) / k)
    functions.append(square)

    # Ensure we have exactly num_functions
    while len(functions) < num_functions:
        # Add more composite waves with random frequencies
        freq = np.random.uniform(0.5, 3)
        phase = np.random.uniform(0, 2*np.pi)
        functions.append(np.sin(freq * x * num_cycles + phase))

    # Convert to numpy array and transpose to get columns
    return np.array(functions).T

def save_to_csv(data, filename='periodic_functions.csv'):
    """Save the generated functions to a CSV file."""
    np.savetxt(filename, data, delimiter=',')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate periodic function data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--points-per-cycle', type=int, default=100,
                      help='Number of points to sample per cycle')
    parser.add_argument('--num-cycles', type=int, default=10,
                      help='Number of cycles to generate for each function')
    parser.add_argument('--num-functions', type=int, default=4,
                      help='Number of different functions to generate')
    parser.add_argument('--output', type=str, default='periodic_functions.csv',
                      help='Output CSV file name')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.points_per_cycle < 10:
        parser.error("points-per-cycle must be at least 10")
    if args.num_cycles < 1:
        parser.error("num-cycles must be at least 1")
    if args.num_functions < 1:
        parser.error("num-functions must be at least 1")
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # Generate periodic functions with specified parameters
    data = generate_periodic_functions(
        points_per_cycle=args.points_per_cycle,
        num_cycles=args.num_cycles,
        num_functions=args.num_functions
    )
    save_to_csv(data, args.output)
    
    total_points = args.points_per_cycle * args.num_cycles
    print(f"\nGenerated {args.num_functions} periodic functions:")
    print(f"- Points per cycle: {args.points_per_cycle}")
    print(f"- Number of cycles: {args.num_cycles}")
    print(f"- Total points: {total_points:,}")
    print(f"- Output shape: {data.shape}")
    print(f"\nData saved to: {args.output}")
