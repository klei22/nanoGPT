import argparse
from collections import Counter
import statistics

def count_characters(file_path):
    """
    Count the occurrences of each character in a given text file, print the results, and provide summary statistics.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            character_counts = Counter(content)
        
        # Display counts
        for char, count in sorted(character_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"'{char}': {count}")
        
        # Summary statistics
        counts = list(character_counts.values())
        max_usage = max(counts)
        min_usage = min(counts)
        mean_usage = statistics.mean(counts)
        median_usage = statistics.median(counts)
        mode_usage = statistics.mode(counts)

        print("\nSummary Statistics:")
        print(f"Max Usage: {max_usage}")
        print(f"Min Usage: {min_usage}")
        print(f"Mean Usage: {mean_usage:.2f}")
        print(f"Median Usage: {median_usage}")
        print(f"Mode Usage: {mode_usage}")

        # Special count for characters appearing 1 to 100 times
        special_counts = {i: 0 for i in range(1, 101)}
        for count in counts:
            if 1 <= count <= 100:
                special_counts[count] += 1
        
        print("\nCounts of characters appearing from 1 to 100 times:")
        for times, count in special_counts.items():
            if count > 0:  # Only display frequencies that occur
                print(f"{times} times: {count}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Count the occurrences of each character in a text file and provide summary statistics.")
    parser.add_argument("file_path", type=str, help="Path to the text file.")
    args = parser.parse_args()

    count_characters(args.file_path)

if __name__ == "__main__":
    main()

