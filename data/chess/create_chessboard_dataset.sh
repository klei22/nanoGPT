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

# Use defaults
python3 chess_utils/filter.py
python3 chess_utils/extract_moveset.py
python3 chess_utils/create_chessboard_input.py
