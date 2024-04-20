#!/bin/bash

python3 get_dataset.py
python3 process_games.py
python3 moves_to_json.py
python3 filter.py parsed_games.json big.txt --min_elo 1500
python3 extract_moveset.py big.txt big_formatted_movesets.txt
python3 test.py big_formatted_movesets.txt big_output.txt

