import json
import pandas as pd
from datasets import load_dataset

# Function to preprocess dialogue and summary by only removing '#' characters
def preprocess_text(row):
    row['dialogue'] = row['dialogue'].replace("#", "")
    row['summary'] = row['summary'].replace("#", "")
    row['topic'] = row['topic'].replace("#", "")
    return row

# Load the DialogSum dataset from Hugging Face
dataset = load_dataset("knkarthick/dialogsum")

# Dataset splits
splits = ['train', 'validation', 'test']

# Initialize an empty list to hold DataFrames
dataframes = []

for split in splits:
    # Convert each split to a pandas DataFrame
    df = pd.DataFrame(dataset[split])

    # Apply preprocessing to each row
    df = df.apply(preprocess_text, axis=1)

    # Append the processed DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a JSON file
combined_df.to_json("processed_combined_dialogsum.json", orient="records", lines=True)

print("Processed and combined JSON file created successfully.")


def json_to_text(json_path, text_path):
    # Open the JSON file and the target text file
    with open(json_path, 'r') as json_file, open(text_path, 'w') as text_file:
        for line in json_file:
            # Load the JSON object from each line
            data = json.loads(line)

            # Write the formatted text to the text file
            text_file.write(f"text:\n{data['dialogue']}\nsummary:\n{data['summary']}\ntopic:\n{data['topic']}\n\n")

# Path to the JSON file
json_path = "processed_combined_dialogsum.json"

# Path to the target text file
text_path = "input.txt"

# Convert JSON to text
json_to_text(json_path, text_path)

print("JSON converted to text file successfully.")

