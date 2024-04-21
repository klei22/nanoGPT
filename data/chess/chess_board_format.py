import chess
import argparse
import re

def print_compact_ascii_board(board):
    # Generate compact ASCII representation of the board without borders and spaces
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            symbol = piece.symbol() if piece else "0"
            print(symbol, end="")
        print()

def resolve_ambiguous_move(board, move):
    # Attempt to resolve moves like "Nbd2" manually
    if len(move) == 4 and move[0] in "NBRQK" and move[1] in "abcdefgh":
        piece_type = {'N': chess.KNIGHT, 'B': chess.BISHOP, 'R': chess.ROOK, 'Q': chess.QUEEN, 'K': chess.KING}[move[0]]
        start_file = 'abcdefgh'.index(move[1])
        end_square = chess.SQUARE_NAMES.index(move[2:])
        candidates = []
        for i in range(8):
            square = chess.square(start_file, i)
            if board.piece_at(square) and board.piece_at(square).piece_type == piece_type and board.piece_at(square).color == board.turn:
                if chess.Move(square, end_square) in board.legal_moves:
                    candidates.append(chess.Move(square, end_square))
        if len(candidates) == 1:
            return candidates[0]
    return None

def apply_moves_and_print_boards(moves):
    board = chess.Board()
    for move in moves.split():
        try:
            board.push_san(move)
            print(f"Move: {move}")
            print_compact_ascii_board(board)
            print()  # Print a blank line for better separation
        except ValueError:
            resolved_move = resolve_ambiguous_move(board, move)
            if resolved_move:
                board.push(resolved_move)
                print(f"Resolved and made move: {move}")
                print_compact_ascii_board(board)
                print()
            else:
                print(f"Skipping invalid or unresolved move: {move}")
                break  # Exit the loop and stop processing the current game


def process_games_from_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                moves = line.strip()
                print(f"Processing game: {line.strip()}")
                apply_moves_and_print_boards(moves)
                print("Moving to next game...\n")

def main():
    parser = argparse.ArgumentParser(description="Process a file of chess games and print each board state in ASCII format.")
    parser.add_argument("filename", type=str, help="The filename containing the chess games, each on a new line.")
    args = parser.parse_args()

    process_games_from_file(args.filename)

if __name__ == "__main__":
    main()

