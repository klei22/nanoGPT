import argparse
import os
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def download_file(url, filename):
    """
    Download a file from a given URL with a progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Failed to download the file completely.")
    else:
        print(f"Downloaded {filename}")


def find_parquet_links(url):
    """
    Find all parquet file links on the given URL.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = [
        ("https://huggingface.co" + a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"].endswith(".parquet?download=true")
    ]
    return links


def emit_translation_contents(
    records,
    output_text_file,
    translation_key,
    language_keys,
    value_prefixes,
    skip_empty,
    require_all_languages,
):
    with open(output_text_file, "a") as f:
        prev_item_written = False
        for item in records:
            translation = item.get(translation_key, {})
            if not isinstance(translation, dict):
                continue

            if require_all_languages:
                missing_language = False
                for lang in language_keys:
                    value = translation.get(lang, "")
                    if value is None:
                        value = ""
                    if skip_empty and value == "":
                        missing_language = True
                        break
                if missing_language:
                    continue

            content_written = False
            for lang, prefix in zip(language_keys, value_prefixes):
                value = translation.get(lang, "")
                if value is None:
                    value = ""
                if skip_empty and value == "":
                    continue
                if prev_item_written or content_written:
                    f.write("\n")
                f.write((prefix + value).strip())
                prev_item_written = True
                content_written = True


def main(
    url,
    output_text_file,
    translation_key,
    language_keys,
    value_prefixes,
    skip_empty,
    append,
    max_files,
    require_all_languages,
):
    if len(language_keys) != len(value_prefixes):
        raise ValueError("language_keys and value_prefixes must have the same length.")

    parquet_links = find_parquet_links(url)

    if max_files is not None:
        parquet_links = parquet_links[:max_files]
        print(f"Limiting to first {len(parquet_links)} parquet files.")

    download_dir = "./downloaded_parquets"
    os.makedirs(download_dir, exist_ok=True)

    if not append:
        open(output_text_file, "w").close()

    for link in parquet_links:
        file_name = link.split("/")[-1].split("?")[0]
        parquet_path = os.path.join(download_dir, file_name)

        if not os.path.exists(parquet_path):
            download_file(link, parquet_path)

        df = pd.read_parquet(parquet_path)
        records = df.to_dict(orient="records")

        emit_translation_contents(
            records,
            output_text_file,
            translation_key,
            language_keys,
            value_prefixes,
            skip_empty,
            require_all_languages,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape and convert Parquet files with nested translation dicts to a text file."
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL to scrape for Parquet files.",
    )
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file where the contents should be saved.",
    )
    parser.add_argument(
        "--translation_key",
        type=str,
        default="translation",
        help="Column name containing the translation dictionary.",
    )
    parser.add_argument(
        "--language_keys",
        type=str,
        nargs="+",
        required=True,
        help="List of language keys inside the translation dictionary to emit.",
    )
    parser.add_argument(
        "-p",
        "--value_prefix",
        type=str,
        nargs="+",
        required=True,
        help="List of prefixes to be added to each language value.",
    )
    parser.add_argument(
        "-s",
        "--skip_empty",
        default=False,
        action="store_true",
        help="Skip any item which is the empty string",
    )
    parser.add_argument(
        "-a",
        "--append",
        default=False,
        action="store_true",
        help="Append to the current output text file",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit on the number of parquet files to download (starting from the first).",
    )
    parser.add_argument(
        "--require_all_languages",
        default=False,
        action="store_true",
        help="Skip entries missing any requested language (after empty filtering).",
    )
    args = parser.parse_args()
    main(
        args.url,
        args.output_text_file,
        args.translation_key,
        args.language_keys,
        args.value_prefix,
        args.skip_empty,
        args.append,
        args.max_files,
        args.require_all_languages,
    )
