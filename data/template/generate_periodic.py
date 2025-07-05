import numpy as np

def generate_periodic_functions(num_points=1000, num_functions=4):
    """Generate different periodic functions with varying frequencies and phases."""
    x = np.linspace(0, 10, num_points)
    functions = []
    
    # Basic sine wave
    functions.append(np.sin(x))
    
    # Cosine wave with different frequency
    functions.append(np.cos(2 * x))
    
    # Composite wave (sum of sines)
    functions.append(0.5 * np.sin(x) + 0.3 * np.sin(3 * x))
    
    # Square wave approximation using Fourier series
    square = np.zeros_like(x)
    for k in range(1, 10, 2):
        square += (4/np.pi) * (np.sin(k * x) / k)
    functions.append(square)
    
    # Ensure we have exactly num_functions
    while len(functions) < num_functions:
        # Add more composite waves with random frequencies
        freq = np.random.uniform(0.5, 3)
        phase = np.random.uniform(0, 2*np.pi)
        functions.append(np.sin(freq * x + phase))
    
    # Convert to numpy array and transpose to get columns
    return np.array(functions).T

def save_to_csv(data, filename='periodic_functions.csv'):
    """Save the generated functions to a CSV file."""
    np.savetxt(filename, data, delimiter=',')

if __name__ == '__main__':
    # Generate 4 different periodic functions with 1000 points each
    data = generate_periodic_functions(num_points=1000, num_functions=4)
    save_to_csv(data)
    print(f"Generated {data.shape[1]} periodic functions with {data.shape[0]} points each.")
    print(f"Data saved to periodic_functions.csv") 