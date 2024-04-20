import os
import argparse
import requests
import zstandard as zstd
from datetime import datetime
import json
import re
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

def download_and_decompress(url, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if the file is already downloaded and decompressed
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
        return

    # Perform the actual download
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    dctx = zstd.ZstdDecompressor()
    console = Console()
    console.log("Starting download...")

    with open(output_path, 'wb') as file:
        with Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            TransferSpeedColumn(),  # Displays the download speed
            TimeRemainingColumn(),  # Estimates the time remaining for the download
            console=console,
            transient=True
        ) as progress:
            download_task = progress.add_task("Downloading", total=total_size, filename=os.path.basename(url))
            for chunk in response.iter_content(chunk_size=16384):
                # Directly write the downloaded chunk to the file
                file.write(chunk)
                progress.update(download_task, advance=len(chunk))

    console.log("Download completed. Starting decompression...")
    # Decompress the file after download completes
    with open(output_path, 'rb') as compressed_file, open(output_path.replace('.zst', '.txt'), 'wb') as decompressed_file, \
         console.status("[bold green]Decompressing...", spinner="dots"):
        decompressed_data = dctx.decompress(compressed_file.read())
        decompressed_file.write(decompressed_data)
    console.log("Decompression completed.")

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

        if not all(key in game_info for key in ["WhiteElo", "BlackElo", "Event", "Result"]):
            discarded += 1
            continue

        move_text = game.split(']')[-1].strip()
        move_text = re.search(r'1\..*', move_text, re.DOTALL)
        if not move_text:
            discarded += 1
            continue
        game_info['game'] = move_text.group(0)

        game_info['termination'] = 'n' if 'normal' in game_info.get("Termination", "").lower() else 't'
        result = game_info["Result"]
        game_info['winner'] = 'w' if result == '1-0' else 'b' if result == '0-1' else 't'

        data.append({
            "termination": game_info['termination'],
            "winner": game_info['winner'],
            "whiteElo": game_info["WhiteElo"],
            "blackElo": game_info["BlackElo"],
            "eventType": game_info["Event"],
            "game": game_info['game']
        })
        successful += 1

    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    console.log(f"Successfully added {successful} games to JSON.")
    console.log(f"Discarded {discarded} incomplete games.")

def setup_argparse():
    parser = argparse.ArgumentParser(description='Download and decompress Lichess dataset')
    default_url = f'https://database.lichess.org/standard/lichess_db_standard_rated_2015-01.pgn.zst'
    parser.add_argument('--url', type=str, default=default_url, help='URL of the file to download')
    return parser.parse_args()

def main():
    args = setup_argparse()
    dataset_dir = 'datasets'
    zst_file_name = 'lichess_games.zst'
    txt_output_path = os.path.join(dataset_dir, zst_file_name)
    json_output_path = os.path.join(dataset_dir, 'lichess_games.json')

    download_and_decompress(args.url, txt_output_path)
    parse_games_to_json(txt_output_path.replace('.zst', '.txt'), json_output_path)
    print("Conversion to JSON completed.")

if __name__ == '__main__':
    main()

