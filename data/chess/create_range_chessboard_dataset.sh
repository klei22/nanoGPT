#!/bin/bash

# Ensure the chess module is installed, if not exit
if ! python3 -c "import chess" &> /dev/null; then
    echo "The 'chess' module is not installed."
    exit 1
else
    echo "The 'chess' module is installed."
fi

# say what line is running
set -x

# Create files
if [ ! -f "datasets/lichess_games.zst" ]; then
  python3 chess_utils/get_dataset.py
fi
if [ ! -f "datasets/lichess_games.txt" ]; then
  python3 chess_utils/process_games.py
fi
if [ ! -f "json/parsed_games.json" ]; then
  python3 chess_utils/moves_to_json.py
fi

# Process into skill buckets
for (( i = 15; i < 2; i++ )); do
  label="${i}_$((i + 1))"
  python3 chess_utils/filter.py -o "filtered_json/filtered_games_${label}.json" --min_elo "${i}00" --max_elo "$((i + 1))00"
  python3 chess_utils/extract_moveset.py --json_file "filtered_json/filtered_games_${label}.json"
  python3 chess_utils/create_chessboard_input.py --output "input_${label}.txt"
done
