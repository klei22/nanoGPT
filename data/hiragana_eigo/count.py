import argparse
from collections import Counter
import statistics
import plotly.graph_objects as go
import plotly.express as px

def count_characters(file_path, plot=False):
    """
    Count the occurrences of each character in a given text file, print the results, and provide summary statistics.
    Optionally generate a plotly graph if 'plot' is True.
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
        print(f"Number of Chars: {max_usage}")
        print(f"Max Usage: {max_usage}")
        print(f"Min Usage: {min_usage}")
        print(f"Mean Usage: {mean_usage:.2f}")
        print(f"Median Usage: {median_usage}")
        print(f"Mode Usage: {mode_usage}")

        # Generate plots if requested
        if plot:
            generate_plots(character_counts)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def generate_plots(character_counts):
    # Character frequency plot
    items = sorted(character_counts.items(), key=lambda item: item[1], reverse=True)
    chars, counts = zip(*items)
    fig = go.Figure([go.Bar(x=list(chars), y=list(counts), text=list(counts), textposition='auto')])
    fig.update_layout(title='Character Frequency', xaxis_title='Characters', yaxis_title='Counts', yaxis_type="log")
    fig.show()

    # Histogram of counts
    fig = px.histogram(x=counts, log_y=True, nbins=50)
    fig.update_layout(title='Histogram of Character Counts', xaxis_title='Number of Appearances', xaxis_type='log', yaxis_title='Frequency')
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Count the occurrences of each character in a text file and provide summary statistics.")
    parser.add_argument("file_path", type=str, help="Path to the text file.")
    parser.add_argument("--plot", action="store_true", help="Generate plots using Plotly.")
    args = parser.parse_args()

    count_characters(args.file_path, args.plot)

if __name__ == "__main__":
    main()

