import argparse

def create_digit_map(offset, num_digits):
    """
    Creates a dictionary that maps digits (0-num_digits) to characters with a specified UTF offset.
    Example: offset = 97 for 'a' -> 'j' mapping (if num_digits = 10).
    """
    digit_map = {}
    for i in range(num_digits):
        digit_map[i] = chr(offset + i)
    return digit_map

def modular_addition(a, b, mod):
    """
    Performs modular addition of two numbers.
    """
    return (a + b) % mod

def represent_number(num, digit_map):
    """
    Represents the number using the digit map provided.
    Example: 123 -> 'abc' if map maps 1->'a', 2->'b', 3->'c'.
    """
    return ''.join(digit_map[int(digit)] for digit in str(num))

def all_modular_additions(mod, offset, num_digits):
    """
    Generates all modular additions for two numbers 0-(num_digits-1), 
    and allows for different representations by changing the UTF-8 offset.

    :param mod: The modulus for the addition.
    :param offset: UTF offset to represent digits.
    :param num_digits: Number of digits to consider in modular arithmetic.
    """
    digit_map = create_digit_map(offset, num_digits)
    for i in range(mod):
        for j in range(mod):
            result = modular_addition(i, j, mod)
            i_rep = represent_number(i, digit_map)
            j_rep = represent_number(j, digit_map)
            result_rep = represent_number(result, digit_map)
            print(f"{i_rep} + {j_rep} (mod {mod}) = {result_rep}")

def main():
    parser = argparse.ArgumentParser(description="Generate modular additions with customizable representations.")
    parser.add_argument("-u", "--unoverlapped-offsets", type=int, default=1, help="Number of unoverlapped offsets.")
    parser.add_argument("-m", "--max-digits", type=int, default=10, help="Max number of digits for modular arithmetic.")
    parser.add_argument("-o", "--start-offset", type=int, default=48, help="Starting UTF offset (default 48 for '0').")

    args = parser.parse_args()

    unoverlapped_offsets = args.unoverlapped_offsets
    max_digits = args.max_digits
    start_offset = args.start_offset

    # Calculate the total required size to ensure non-overlapping ranges
    required_size = max_digits * unoverlapped_offsets

    # Ensure we can pack all representations within the available UTF-8 space
    print(f"Generating {unoverlapped_offsets} non-overlapping representations with modulus {max_digits}:")

    for rep in range(unoverlapped_offsets):
        current_offset = start_offset + (rep * max_digits)
        print(f"\nRepresentation {rep + 1} (offset {current_offset}):")
        all_modular_additions(max_digits, current_offset, max_digits)

if __name__ == "__main__":
    main()

