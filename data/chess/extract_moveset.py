import re
import json
import argparse

def extract_movesets(json_file, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    with open(output_file, 'w') as file:
        for entry in data:
            if 'Moveset' in entry:
                moveset = entry['Moveset']
                if not any(char in moveset for char in ['%', '{', '}', '[', ']', '?']):
                    cleaned_moveset = re.sub(r'\s(1-0|0-1|1/2-1/2)$', '', moveset)
                    file.write(cleaned_moveset + '\n')

def main():
    parser = argparse.ArgumentParser(description="Extract and clean movesets from JSON file.")
    parser.add_argument('json_file', type=str, help='Path to the JSON file')
    parser.add_argument('output_file', type=str, help='Path to the output text file where cleaned movesets will be saved')
    args = parser.parse_args()
    extract_movesets(args.json_file, args.output_file)

if __name__ == '__main__':
    main()
