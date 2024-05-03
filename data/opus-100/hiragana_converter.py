import argparse
from yakinori import Yakinori

def parse_args():
    parser = argparse.ArgumentParser(description="Convert sentences in a file to hiragana.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    return parser.parse_args()

def main():
    args = parse_args()
    yakinori = Yakinori()

    with open(args.input_file, "r", encoding="utf-8") as input_file:
        with open(args.output_file, "w", encoding="utf-8") as output_file:
            for line in input_file:
                line = line.strip()
                parsed_list = yakinori.get_parsed_list(line)
                hiragana_sentence = yakinori.get_hiragana_sentence(parsed_list)
                output_file.write(hiragana_sentence + "\n")

if __name__ == "__main__":
    main()

