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
                words = line.rstrip().split()  # Split the line into words to maintain spaces
                hiragana_words = []
                for word in words:
                    parsed_list = yakinori.get_parsed_list(word)
                    hiragana_sentence = yakinori.get_hiragana_sentence(parsed_list)
                    hiragana_words.append(hiragana_sentence)
                output_file.write(' '.join(hiragana_words) + "\n")  # Re-join words with a space

if __name__ == "__main__":
    main()

