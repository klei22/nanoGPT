import os
import argparse
import requests
import zstandard as zstd
from tqdm import tqdm
from datetime import datetime
import json
import re
from rich.console import Console
from rich.progress import Progress
import re

def download_and_decompress(url, output_path):
    # Check if the file is already downloaded and decompressed
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
        return
    # Perform the actual download and decompression
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    dctx = zstd.ZstdDecompressor()
    with open(output_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading and Decompressing') as progress:
            with dctx.stream_reader(response.raw) as decompressor:
                while True:
                    chunk = decompressor.read(16384)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress.update(16384)

def parse_games_to_json(input_file, output_json):
    console = Console()
    with open(input_file, 'r') as file:
        lines = file.readlines()

    games = ''.join(lines).strip().split("\n\n")
    data = []
    successful = 0
    discarded = 0

    for game in games:
        game_info = {}
        headers = re.findall(r"\[([^]]+)\]", game)
        for header in headers:
            key, value = header.split(" ", 1)
            game_info[key.strip()] = value.strip('"')

        # Check if all necessary fields are present
        if not all(key in game_info for key in ["WhiteElo", "BlackElo", "Event", "Result"]):
            discarded += 1
            continue

        # Extract game moves; assuming they start after the last header and begin with '1.'
        move_text = game.split(']')[-1].strip()
        move_text = re.search(r'1\..*', move_text, re.DOTALL)
        if not move_text:
            discarded += 1
            continue
        game_info['game'] = move_text.group(0)

        # Extract and set termination type
        game_info['termination'] = 'n' if 'normal' in game_info.get("Termination", "").lower() else 't'

        # Extract and set winner type
        result = game_info["Result"]
        if result == '1-0':
            game_info['winner'] = 'w'
        elif result == '0-1':
            game_info['winner'] = 'b'
        else:
            game_info['winner'] = 't'

        data.append({
            "termination": game_info['termination'],
            "winner": game_info['winner'],
            "whiteElo": game_info["WhiteElo"],
            "blackElo": game_info["BlackElo"],
            "eventType": game_info["Event"],
            "game": game_info['game']
        })
        successful += 1

    # Write to JSON file
    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    console.log(f"Successfully added {successful} games to JSON.")
    console.log(f"Discarded {discarded} incomplete games.")


def setup_argparse():
    parser = argparse.ArgumentParser(description='Download and decompress Lichess dataset')
    current_year = datetime.now().year
    default_url = f'https://database.lichess.org/standard/lichess_db_standard_rated_2016-01.pgn.zst'
    parser.add_argument('--url', type=str, default=default_url, help='URL of the file to download')
    return parser.parse_args()

def main():
    args = setup_argparse()
    txt_output_path = 'lichess_games.txt'
    json_output_path = 'lichess_games.json'

    download_and_decompress(args.url, txt_output_path)
    parse_games_to_json(txt_output_path, json_output_path)
    print("Conversion to JSON completed.")

if __name__ == '__main__':
    main()

