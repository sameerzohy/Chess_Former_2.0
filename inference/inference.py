"""
ChessFormer Inference Module

This module handles loading the trained ChessFormer model and making predictions
for chess positions. It provides a clean interface for the GUI to interact with.
"""

import torch
import chess
import sys
import os

# Add parent directory to path to import model files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ChessFormer
from dataset import combine_features, PIECE_TO_IDX


class ChessFormerInference:
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize the ChessFormer inference engine.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (.pt file)
            device: torch device to use (cuda/cpu). Auto-detects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading ChessFormer on device: {self.device}")
        
        # Initialize model
        self.model = ChessFormer().to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print("Model loaded successfully!")
        
    def _board_to_model_input(self, board_history):
        """
        Convert a board history to model input tensor.
        
        Args:
            board_history: List of chess.Board objects representing position history
            
        Returns:
            torch.Tensor of shape (1, 64, 112) ready for model input
        """
        raw_feats = combine_features(board_history)  # (64, 112)
        x = raw_feats.unsqueeze(0)  # (1, 64, 112)
        return x.to(self.device)
    
    @staticmethod
    def _model_idx_to_chess_square(model_idx):
        """
        Convert model square index (0-63, a8..h1) to chess.Square (0-63, a1..h8).
        """
        model_rank = model_idx // 8  # 0..7 (0=rank 8, 7=rank 1)
        file = model_idx % 8          # 0..7 (a..h)
        chess_rank = 7 - model_rank   # flip: 0=rank 1, 7=rank 8
        return chess.square(file, chess_rank)
    
    @staticmethod
    def _chess_square_to_model_idx(chess_square):
        """
        Convert chess.Square (a1..h8) to model index (a8..h1).
        """
        file = chess.square_file(chess_square)  # 0..7
        rank = chess.square_rank(chess_square)  # 0..7 (0=rank 1)
        model_rank = 7 - rank
        return model_rank * 8 + file
    
    def predict_move(self, board_history, top_k=5):
        """
        Predict the best moves for the current position.
        
        Args:
            board_history: List of chess.Board objects (current position is last)
            top_k: Number of top moves to return
            
        Returns:
            List of tuples (move, probability) sorted by probability (highest first)
        """
        if len(board_history) == 0:
            raise ValueError("Board history must contain at least one position")
            
        current_board = board_history[-1]
        
        # Get model input
        x = self._board_to_model_input(board_history)
        
        # Get model predictions
        with torch.no_grad():
            policy_logits, promo_logits, value_logits = self.model(x)
            
        # Process policy logits
        policy_logits_flat = policy_logits.view(-1)  # (4096,)
        policy_probs = torch.softmax(policy_logits_flat, dim=0)
        
        # Get top-k moves
        top_probs, top_indices = torch.topk(policy_probs, min(top_k * 10, len(policy_probs)))
        
        # Convert to chess moves and filter legal moves
        legal_moves = []
        for idx, prob in zip(top_indices, top_probs):
            idx = idx.item()
            prob = prob.item()
            
            from_model_idx = idx // 64
            to_model_idx = idx % 64
            
            from_square = self._model_idx_to_chess_square(from_model_idx)
            to_square = self._model_idx_to_chess_square(to_model_idx)
            
            # Check all possible promotions for this move
            base_move = chess.Move(from_square, to_square)
            
            # If it's a pawn promotion move
            piece = current_board.piece_at(from_square)
            if (piece and piece.piece_type == chess.PAWN and 
                chess.square_rank(to_square) in [0, 7]):
                
                # Get promotion probabilities
                from_feats_idx = from_model_idx
                promo_probs = torch.softmax(promo_logits[0, from_feats_idx, :], dim=0)
                
                # Try promotions in order of probability
                promo_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                promo_indices = [1, 2, 3, 4]  # indices in promo_logits
                
                for promo_type, promo_idx in zip(promo_types, promo_indices):
                    move = chess.Move(from_square, to_square, promotion=promo_type)
                    if move in current_board.legal_moves:
                        # Combine policy and promotion probabilities
                        combined_prob = prob * promo_probs[promo_idx].item()
                        legal_moves.append((move, combined_prob))
            else:
                # Regular move (no promotion)
                if base_move in current_board.legal_moves:
                    legal_moves.append((base_move, prob))
            
            if len(legal_moves) >= top_k:
                break
        
        # Sort by probability and return top-k
        legal_moves.sort(key=lambda x: x[1], reverse=True)
        return legal_moves[:top_k]
    
    def predict_value(self, board_history):
        """
        Predict the value (win/draw/loss) for the current position.
        
        Args:
            board_history: List of chess.Board objects (current position is last)
            
        Returns:
            Dictionary with probabilities: {'loss': float, 'draw': float, 'win': float}
        """
        x = self._board_to_model_input(board_history)
        
        with torch.no_grad():
            _, _, value_logits = self.model(x)
            
        value_probs = torch.softmax(value_logits[0], dim=0).cpu().numpy()
        
        return {
            'loss': float(value_probs[0]),
            'draw': float(value_probs[1]),
            'win': float(value_probs[2])
        }
    
    def get_best_move(self, board_history):
        """
        Get the single best move for the current position.
        
        Args:
            board_history: List of chess.Board objects (current position is last)
            
        Returns:
            chess.Move object representing the best move
        """
        moves = self.predict_move(board_history, top_k=1)
        if moves:
            return moves[0][0]  # Return just the move, not the probability
        return None


def main():
    """Example usage of the inference engine."""
    # Path to the checkpoint
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoint", "best_chessformer.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Initialize inference engine
    engine = ChessFormerInference(checkpoint_path)
    
    # Create a test position
    board = chess.Board()
    board_history = [board.copy()]
    
    # Make some moves
    test_moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    for move_uci in test_moves:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        board_history.append(board.copy())
        
    print(f"\nCurrent position:\n{board}\n")
    
    # Get top moves
    print("Top 5 predicted moves:")
    top_moves = engine.predict_move(board_history, top_k=5)
    for i, (move, prob) in enumerate(top_moves, 1):
        print(f"{i}. {move.uci()} (probability: {prob:.4f})")
    
    # Get position evaluation
    print("\nPosition evaluation:")
    value = engine.predict_value(board_history)
    print(f"Win: {value['win']:.2%}, Draw: {value['draw']:.2%}, Loss: {value['loss']:.2%}")


if __name__ == "__main__":
    main()
