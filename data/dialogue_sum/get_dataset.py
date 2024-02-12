import pandas as pd
from datasets import load_dataset

# Load the DialogSum dataset from Hugging Face
dataset = load_dataset("knkarthick/dialogsum")

# DialogSum dataset splits
splits = ['train', 'validation', 'test']

# Initialize an empty list to hold DataFrames
dataframes = []

for split in splits:
    # Convert each split to a pandas DataFrame
    df = pd.DataFrame(dataset[split])

    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv("combined_dialogsum.csv", index=False)

print("Combined CSV file created successfully.")

