import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Remove contiguous pairs of lines containing specified characters from a file, and separate contiguous sets with a newline.")
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
        lines = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        i = 0
        skip_next = False
        progress_bar = tqdm(total=len(lines), desc="Processing", unit="line")
        group_buffer = []  # Buffer to hold lines in a contiguous group

        while i < len(lines):
            # Check for end of file or single empty line as separator
            if i + 1 < len(lines) and lines[i+1] == '\n' and (i + 2 == len(lines) or lines[i+2] != '\n'):
                next_line = i + 2  # Skip over the blank line
            else:
                next_line = i + 1

            if next_line < len(lines) and (any(c in char_set for c in lines[i]) or any(c in char_set for c in lines[next_line])):
                if not skip_next and group_buffer:
                    # Output the buffered group followed by a newline
                    outfile.writelines(group_buffer)
                    outfile.write("\n")
                    group_buffer = []
                skip_next = True  # Mark to skip the next line too
            else:
                if not skip_next:
                    group_buffer.append(lines[i])  # Buffer the line if not part of a skip

            i = next_line  # Move to the next line or skip a blank separator
            progress_bar.update(i - progress_bar.n)  # Update progress bar to the new index

        if group_buffer:  # Check if there's anything left in the buffer after the last line
            outfile.writelines(group_buffer)
            outfile.write("\n")  # Make sure to output the last group of lines

        progress_bar.close()  # Ensure the progress bar is properly closed after the loop

def main():
    args = parse_args()
    remove_contiguous_lines(args.input_file, args.output_file, args.chars_file)

if __name__ == "__main__":
    main()

