import argparse

def write_numbers_to_file(n):
    with open('input.txt', 'w') as file:
        for number in range(1, n + 1):
            file.write(f"{number}\n")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Generate a file with numbers from 1 to n.')

    # Add the 'n' argument
    parser.add_argument('n', type=int, help='The upper limit number')

    # Parse the arguments
    args = parser.parse_args()

    # Generate the file
    write_numbers_to_file(args.n)

