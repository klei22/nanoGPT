import argparse
from collections import Counter

def count_characters(file_path):
    """
    Count the occurrences of each character in a given text file and print the results.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            character_counts = Counter(content)
        
        # Sort characters by frequency (descending)
        for char, count in sorted(character_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"'{char}': {count}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Count the occurrences of each character in a text file.")
    parser.add_argument("file_path", type=str, help="Path to the text file.")
    args = parser.parse_args()

    count_characters(args.file_path)

if __name__ == "__main__":
    main()

