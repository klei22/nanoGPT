import json
import argparse

def extract_movesets(json_file, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    with open(output_file, 'w') as file:
        for entry in data:
            if 'Moveset' in entry:
                file.write(entry['Moveset'] + '\n')

def main():
    parser = argparse.ArgumentParser(description="Extract Movesets from JSON file.")
    parser.add_argument('json_file', type=str, help='Path to the JSON file')
    parser.add_argument('output_file', type=str, help='Path to the output text file where Movesets will be saved')

    args = parser.parse_args()
    extract_movesets(args.json_file, args.output_file)

if __name__ == '__main__':
    main()
