import argparse
from pathlib import Path

TOKENS_FILE = "tokensfile.txt"


def emit_tokenlist(tokens):
    """
    Write tokens (iterable of single-char strings) to TOKENS_FILE.
    """
    Path(TOKENS_FILE).write_text("".join(tokens), encoding="utf-8")


def transform_lowercase(text):
    """
    Convert text to lowercase while preserving non-letter characters.
    """
    transformed = text.lower()
    emit_tokenlist(sorted(set(transformed)))
    return transformed


def transform_case_map(text):
    """
    Map each character:
      - 'L' if lowercase
      - 'U' if uppercase
      - '_' for everything else
    """
    transformed_chars = []
    for char in text:
        if char.islower():
            transformed_chars.append("L")
        elif char.isupper():
            transformed_chars.append("U")
        else:
            transformed_chars.append("_")
    emit_tokenlist(["L", "U", "_"])
    return "".join(transformed_chars)


def transform_file(filename, method):
    """
    Transforms a file in-place using the selected method.
    """
    try:
        with open(filename, "r+", encoding="utf-8") as file:
            file_content = file.read()

            if method == "lowercase":
                transformed_content = transform_lowercase(file_content)
            elif method == "case_map":
                transformed_content = transform_case_map(file_content)
            else:
                raise ValueError(f"Unknown method: {method}")

            file.seek(0)
            file.write(transformed_content)
            file.truncate()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a text file by replacing characters.")
    parser.add_argument("input_file", help="The input text file to transform.")
    parser.add_argument(
        "--method",
        choices=["lowercase", "case_map"],
        default="lowercase",
        help="Which transformation method to use.",
    )
    args = parser.parse_args()
    transform_file(args.input_file, args.method)
