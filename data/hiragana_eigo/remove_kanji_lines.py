import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Filter out groups of contiguous lines containing specified characters from a file.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("chars_file", help="Path to the file containing characters to filter")
    return parser.parse_args()

def remove_contiguous_lines(input_file, output_file, chars_file):
    # Read characters to filter from the file
    with open(chars_file, 'r', encoding='utf-8') as f:
        chars = f.read().strip()
    char_set = set(chars)  # Use a set for faster membership testing

    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    groups = content.split('\n\n')  # Split content into groups separated by at least two newlines

    with open(output_file, 'w', encoding='utf-8') as outfile:
        progress_bar = tqdm(total=len(groups), desc="Processing", unit="group")
        for group in groups:
            lines = group.split('\n')  # Split the group into individual lines
            if any(any(char in char_set for char in line) for line in lines):
                progress_bar.update(1)
                continue  # Skip this group if any line contains a forbidden character

            # Write group followed by a double newline for clarity between groups
            outfile.write(group + '\n\n')
            progress_bar.update(1)

        progress_bar.close()  # Ensure the progress bar is properly closed after processing

def main():
    args = parse_args()
    remove_contiguous_lines(args.input_file, args.output_file, args.chars_file)

if __name__ == "__main__":
    main()

