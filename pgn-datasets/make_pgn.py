import chess.pgn
import random 
import hashlib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_FILES = [
    os.path.join(script_dir, "lichess_elite_2021-07.pgn"),
    os.path.join(script_dir, "lichess_elite_2021-09.pgn")
]

OUTPUT_FILES = os.path.join(script_dir, "final_800k_elite.pgn")
MAX_GAMES = 800_000
MIN_PILES = 10 

print('Scanning games...')

game_pointers = []
seen_hashes = set() 

def game_hash(game):
    moves = " ".join(str(m) for m in game.mainline_moves())
    return hashlib.sha256(moves.encode('utf-8')).hexdigest()

for file_path in INPUT_FILES:
    if len(game_pointers) >= MAX_GAMES:
        break
    try:
        with open(file_path, "r", encoding="utf-8") as f: 
            while len(game_pointers) < MAX_GAMES: 
                offset = f.tell()
                game = chess.pgn.read_game(f)
                
                if len(game_pointers) % 1000 == 0:
                    print(len(game_pointers))
                    
                if game is None: 
                    break 
                
                if len(list(game.mainline_moves())) < MIN_PILES:
                    continue 
        
                h = game_hash(game)
                if h in seen_hashes:
                    continue 
                seen_hashes.add(h)
                game_pointers.append((file_path, offset))
    except FileNotFoundError:
        print(f"Warning: Input file not found: {file_path}")


print(f"Found {len(game_pointers)} games. Shuffling and writing...")
random.shuffle(game_pointers)

#writing a ouput pgn files:
with open(OUTPUT_FILES, "w", encoding="utf-8") as f_out: 
    for file_path, offset in game_pointers:
        with open(file_path, "r", encoding="utf-8") as f_in:
            f_in.seek(offset)
            game = chess.pgn.read_game(f_in)
            if game:
                exporter = chess.pgn.FileExporter(f_out)
                game.accept(exporter)

print("Done.")