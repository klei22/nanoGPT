import os
import argparse
import csv
import json

def tsv_to_json(input_file, output_file):
    lists = []
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        next(reader) # skip the title row
        for row in reader:
            data = {
                "path": row['path'],
                "sentence": row['sentence'],
                "up_votes": row['up_votes'],
                "down_votes": row['down_votes'],
                "age": row['age'],
                "gender": row['gender'],
                "accents": row['accents'],
                "variant": row['variant'],
                "locale": row['locale'],
                "segment": row['segment']
            }
            lists.append(data)

    with open(output_file, 'w') as json_file:
        json.dump(lists, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide a large text file into smaller files.")
    parser.add_argument("input_file", type=str, help="Path to the input tsv file.")
    parser.add_argument("output_json", type=str, help="Path to the output json file.")

    args = parser.parse_args()
    tsv_to_json(args.input_file, args.output_json)
