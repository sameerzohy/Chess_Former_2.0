import torch
from torch.utils.data import Dataset
import chess
import chess.pgn
import os 

NUM_PIECE_TYPES = 12
HISTORY_STEPS = 8
PIECE_HIST_DIM = HISTORY_STEPS * NUM_PIECE_TYPES
META_DIM = 16
RAW_FEAT_DIM = PIECE_HIST_DIM + META_DIM 
MODEL_DIM = 256

class PGN_Moves(Dataset):
    def __init__(self, file_path, max_games=None, max_moves=None):
        """
        file_path: path to .pgn file
        max_games: optional limit on number of games to parse
        max_moves: optional limit on total moves/samples
        """
        self.samples = []

        num_games_parsed = 0

        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                if max_games is not None and num_games_parsed >= max_games:
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break  # EOF

                num_games_parsed += 1

                result = game.headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue  # skip unfinished or weird results

                board = game.board()       # initial position
                board_history = [board.copy()]

                for move in game.mainline_moves():
                    # Build input features for current position (before move)
                    raw_feats = combine_features(board_history)  # (64,112)

                    # Map move from python-chess square indexing -> model indexing
                    raw_from = move.from_square  # 0..63 (a1..h8)
                    raw_to = move.to_square

                    from_sq = self.sq_to_model_idx(raw_from)
                    to_sq = self.sq_to_model_idx(raw_to)

                    # Promotion id
                    promo_id = 0
                    if move.promotion is not None:
                        if move.promotion == chess.QUEEN:
                            promo_id = 1
                        elif move.promotion == chess.ROOK:
                            promo_id = 2
                        elif move.promotion == chess.BISHOP:
                            promo_id = 3
                        elif move.promotion == chess.KNIGHT:
                            promo_id = 4

                    # Value label from POV of side to move
                    value_label = self.result_to_value_label(result, board.turn)

                    # Store sample
                    self.samples.append({
                        "x": raw_feats,          # (64,112) float32
                        "from_sq": from_sq,      # int 0..63
                        "to_sq": to_sq,          # int 0..63
                        "promo": promo_id,       # int 0..4
                        "value": value_label,    # int 0..2
                    })

                    if max_moves is not None and len(self.samples) >= max_moves:
                        break

                    # Apply the move and extend history
                    board.push(move)
                    board_history.append(board.copy())

                if max_moves is not None and len(self.samples) >= max_moves:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # x is already a float32 tensor of shape (64, 112)
        x = sample["x"]
        from_sq = torch.tensor(sample["from_sq"], dtype=torch.long)
        to_sq = torch.tensor(sample["to_sq"], dtype=torch.long)
        promo = torch.tensor(sample["promo"], dtype=torch.long)
        value = torch.tensor(sample["value"], dtype=torch.long)
        return x, from_sq, to_sq, promo, value

    @staticmethod
    def sq_to_model_idx(sq: int) -> int:
        file = sq % 8       # 0..7
        rank = sq // 8      # 0..7 (0 = rank 1)
        model_rank = 7 - rank
        return model_rank * 8 + file  # 0..63 (a8..h1)

    @staticmethod
    def result_to_value_label(result_str: str, side_to_move: bool) -> int:
        """
        result_str: "1-0", "0-1", "1/2-1/2"
        side_to_move: True for white, False for black
        returns: 0=loss, 1=draw, 2=win for side_to_move
        """
        if result_str == "1/2-1/2":
            return 1  # draw

        if result_str == "1-0":
            # White won
            return 2 if side_to_move == chess.WHITE else 0
        elif result_str == "0-1":
            # Black won
            return 2 if side_to_move == chess.BLACK else 0

        # Should not reach here with filtered results
        return 1


PIECE_TO_IDX = {
    'P': 0, 
    'N': 1,
    'B': 2,
    'R': 3, 
    'Q': 4,
    'K': 5,
    'p': 6,
    'n': 7,
    'b': 8,
    'r': 9,
    'q': 10,
    'k': 11
}

def board_to_piece_fast(board: chess.Board):
    """
    Returns an (8, 8) grid of piece indices, -1 = empty.
    Row 0 corresponds to rank 8 (a8..h8), row 7 to rank 1 (a1..h1),
    matching the original board_to_piece orientation.
    """
    grid = torch.full((8, 8), -1, dtype=torch.long)

    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)   # 0..7, 0 = rank 1
        file = chess.square_file(sq)   # 0..7, 0 = file a

        # row 0 = rank 8, so flip the rank
        row = 7 - rank
        col = file

        grid[row, col] = PIECE_TO_IDX[piece.symbol()]

    return grid


def build_piece_history(board_history):
    """
    Builds the (64, HISTORY_STEPS * NUM_PIECE_TYPES) one-hot piece history
    in a vectorized way, preserving the original semantics:
    - If history is shorter than H, it is left-padded with the first board.
    - If longer than H, keep only the last H boards.
    """
    H = HISTORY_STEPS
    if len(board_history) == 0:
        raise ValueError("board history must have atleast 1 board")

    # Ensure exactly H history steps, same as original logic
    if len(board_history) < H:
        first = board_history[0]
        board_history = [first] * (H - len(board_history)) + board_history
    else:
        board_history = board_history[-H:]

    # piece_grids: (H, 8, 8), entries in {-1, 0..11}
    piece_grids = torch.stack(
        [board_to_piece_fast(board) for board in board_history],
        dim=0
    )  # (H, 8, 8)

    # Flatten spatial dims: (H, 64)
    flat = piece_grids.view(H, 64)  # (H, 64)

    # Mask where there is a piece
    mask = flat >= 0
    if not mask.any():
        # no pieces (shouldn't really happen), return all zeros
        return torch.zeros((64, H * NUM_PIECE_TYPES), dtype=torch.float32)

    # Indices of filled squares
    pos = mask.nonzero(as_tuple=False)  # (K, 2): [t, square]
    t_idx = pos[:, 0]                   # time indices
    s_idx = pos[:, 1]                   # square indices
    piece_idx = flat[mask]              # (K,)

    # Build one-hot feature: (H, 64, 12)
    feat = torch.zeros((H, 64, NUM_PIECE_TYPES), dtype=torch.float32)
    feat[t_idx, s_idx, piece_idx] = 1.0

    # Reshape to (64, H*12), with history blocks per square
    feat = feat.permute(1, 0, 2).contiguous()  # (64, H, 12)
    feat = feat.view(64, H * NUM_PIECE_TYPES)  # (64,  H*12)

    return feat 


def build_metadata_features(board_history):
    current = board_history[-1]
    meta = torch.zeros(META_DIM, dtype=torch.float32)

    # Side to move
    meta[0] = 1.0 if current.turn == chess.WHITE else 0.0 
    
    # Castling rights
    castling_str = current.castling_xfen()
    meta[1] = 1.0 if 'K' in castling_str else 0.0 
    meta[2] = 1.0 if 'Q' in castling_str else 0.0 
    meta[3] = 1.0 if 'k' in castling_str else 0.0 
    meta[4] = 1.0 if 'q' in castling_str else 0.0
    
    # En-passant possibility
    meta[5] = 0.0 if current.ep_square is None else 1.0 

    # Halfmove clock / fullmove number (normalized)
    meta[6] = min(current.halfmove_clock / 100.0, 1.0)
    meta[7] = min(current.fullmove_number / 200.0, 1.0)
    
    # Repetition-like features over last H plies
    H = min(len(board_history), HISTORY_STEPS)
    seen = set()
    for i in range(H):
        b = board_history[-H + i]
        key = (b.board_fen(), b.turn, b.castling_xfen(), b.ep_square)
        meta[8 + i] = 1.0 if key in seen else 0.0
        seen.add(key)

    return meta 


def combine_features(board_history):
    piece_history = build_piece_history(board_history)   # (64, PIECE_HIST_DIM)
    meta = build_metadata_features(board_history)        # (META_DIM,)

    meta_broadcast = meta.unsqueeze(0).expand(64, -1)    # (64, META_DIM)
    raw_feats = torch.cat([piece_history, meta_broadcast], dim=1)  # (64, 112)
    return raw_feats