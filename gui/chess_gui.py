"""
ChessFormer GUI - Interactive Chess Board with AI Predictions

A beautiful and interactive chess GUI using Tkinter that integrates with
the ChessFormer model to show AI move predictions and evaluations.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import chess
import chess.svg
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference import ChessFormerInference


# Unicode chess pieces
PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

# Colors
LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE = "#B58863"
HIGHLIGHT_COLOR = "#FFD700"
MOVE_HINT_COLOR = "#90EE90"
SELECTED_COLOR = "#FFFF00"


class ChessGUI:
    def __init__(self, root, checkpoint_path=None):
        self.root = root
        self.root.title("ChessFormer - AI Chess Assistant")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Chess state
        self.board = chess.Board()
        self.board_history = [self.board.copy()]
        self.selected_square = None
        self.legal_moves_from_selected = []
        self.move_history = []
        
        # AI engine
        self.engine = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.engine = ChessFormerInference(checkpoint_path)
                self.ai_loaded = True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.ai_loaded = False
        else:
            self.ai_loaded = False
        
        # Setup UI
        self.setup_ui()
        self.draw_board()
        
        if self.ai_loaded:
            self.update_ai_analysis()
    
    def setup_ui(self):
        """Create the main UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Chess board
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Board title
        title_label = ttk.Label(left_frame, text="ChessFormer AI", 
                               font=("Helvetica", 18, "bold"))
        title_label.pack(pady=5)
        
        # Canvas for chess board
        board_container = ttk.Frame(left_frame)
        board_container.pack(pady=10)
        
        self.canvas = tk.Canvas(board_container, width=640, height=640, 
                               bg="white", highlightthickness=2, 
                               highlightbackground="#333333")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_square_click)
        
        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="New Game", 
                  command=self.new_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Undo Move", 
                  command=self.undo_move).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="AI Move", 
                  command=self.make_ai_move).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Flip Board", 
                  command=self.flip_board).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Information and moves
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Status
        status_frame = ttk.LabelFrame(right_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="White to move", 
                                      font=("Helvetica", 12))
        self.status_label.pack()
        
        # Position evaluation
        eval_frame = ttk.LabelFrame(right_frame, text="Position Evaluation", 
                                    padding=10)
        eval_frame.pack(fill=tk.X, pady=5)
        
        self.eval_label = ttk.Label(eval_frame, text="Loading...", 
                                    font=("Helvetica", 10))
        self.eval_label.pack()
        
        if self.ai_loaded:
            self.eval_progress_frame = ttk.Frame(eval_frame)
            self.eval_progress_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(self.eval_progress_frame, text="Win:").grid(row=0, column=0, sticky=tk.W)
            self.win_bar = ttk.Progressbar(self.eval_progress_frame, length=200, 
                                          mode='determinate')
            self.win_bar.grid(row=0, column=1, padx=5)
            self.win_label = ttk.Label(self.eval_progress_frame, text="0%")
            self.win_label.grid(row=0, column=2)
            
            ttk.Label(self.eval_progress_frame, text="Draw:").grid(row=1, column=0, sticky=tk.W)
            self.draw_bar = ttk.Progressbar(self.eval_progress_frame, length=200, 
                                           mode='determinate')
            self.draw_bar.grid(row=1, column=1, padx=5)
            self.draw_label = ttk.Label(self.eval_progress_frame, text="0%")
            self.draw_label.grid(row=1, column=2)
            
            ttk.Label(self.eval_progress_frame, text="Loss:").grid(row=2, column=0, sticky=tk.W)
            self.loss_bar = ttk.Progressbar(self.eval_progress_frame, length=200, 
                                           mode='determinate')
            self.loss_bar.grid(row=2, column=1, padx=5)
            self.loss_label = ttk.Label(self.eval_progress_frame, text="0%")
            self.loss_label.grid(row=2, column=2)
        
        # Top AI moves
        moves_frame = ttk.LabelFrame(right_frame, text="AI Suggested Moves", 
                                     padding=10)
        moves_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.moves_text = scrolledtext.ScrolledText(moves_frame, height=10, 
                                                    font=("Courier", 10), 
                                                    state=tk.DISABLED)
        self.moves_text.pack(fill=tk.BOTH, expand=True)
        
        # Move history
        history_frame = ttk.LabelFrame(right_frame, text="Move History", 
                                       padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, height=8, 
                                                     font=("Courier", 10), 
                                                     state=tk.DISABLED)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        # Board orientation
        self.flipped = False
    
    def draw_board(self):
        """Draw the chess board and pieces."""
        self.canvas.delete("all")
        square_size = 80
        
        for rank in range(8):
            for file in range(8):
                # Determine display coordinates
                if self.flipped:
                    x = file * square_size
                    y = rank * square_size
                    actual_rank = rank
                    actual_file = file
                else:
                    x = file * square_size
                    y = (7 - rank) * square_size
                    actual_rank = rank
                    actual_file = file
                
                # Determine square index (a1=0, h8=63 in python-chess)
                square = chess.square(actual_file, actual_rank)
                
                # Square color
                is_light = (rank + file) % 2 == 0
                color = LIGHT_SQUARE if is_light else DARK_SQUARE
                
                # Highlight selected square
                if square == self.selected_square:
                    color = SELECTED_COLOR
                # Highlight legal move destinations
                elif self.selected_square is not None:
                    for move in self.legal_moves_from_selected:
                        if move.to_square == square:
                            color = MOVE_HINT_COLOR
                            break
                
                # Draw square
                self.canvas.create_rectangle(x, y, x + square_size, y + square_size, 
                                            fill=color, outline="")
                
                # Draw coordinates
                if file == 0:
                    rank_label = str(rank + 1)
                    self.canvas.create_text(x + 5, y + square_size - 5, 
                                          text=rank_label, font=("Arial", 10, "bold"),
                                          fill="#555555", anchor=tk.SW)
                if rank == 0:
                    file_label = chr(ord('a') + file)
                    self.canvas.create_text(x + square_size - 5, y + square_size - 5, 
                                          text=file_label, font=("Arial", 10, "bold"),
                                          fill="#555555", anchor=tk.SE)
                
                # Draw piece
                piece = self.board.piece_at(square)
                if piece:
                    symbol = PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())
                    self.canvas.create_text(x + square_size/2, y + square_size/2, 
                                          text=symbol, font=("Arial", 48), 
                                          fill="black" if piece.color == chess.WHITE else "#444444")
    
    def on_square_click(self, event):
        """Handle square click events."""
        square_size = 80
        file = event.x // square_size
        rank = event.y // square_size
        
        if self.flipped:
            actual_file = file
            actual_rank = rank
        else:
            actual_file = file
            actual_rank = 7 - rank
        
        # Ensure valid square
        if not (0 <= actual_file < 8 and 0 <= actual_rank < 8):
            return
        
        clicked_square = chess.square(actual_file, actual_rank)
        
        # If no square selected, select this square if it has a piece
        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.legal_moves_from_selected = [
                    move for move in self.board.legal_moves 
                    if move.from_square == clicked_square
                ]
                self.draw_board()
        else:
            # Try to make a move
            move_made = False
            for move in self.legal_moves_from_selected:
                if move.to_square == clicked_square:
                    # Handle promotion
                    if move.promotion:
                        # For simplicity, always promote to queen
                        # You could add a dialog here for user choice
                        move = chess.Move(move.from_square, move.to_square, 
                                        promotion=chess.QUEEN)
                    
                    if move in self.board.legal_moves:
                        self.make_move(move)
                        move_made = True
                        break
            
            # Deselect or select new piece
            if not move_made:
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = clicked_square
                    self.legal_moves_from_selected = [
                        move for move in self.board.legal_moves 
                        if move.from_square == clicked_square
                    ]
                else:
                    self.selected_square = None
                    self.legal_moves_from_selected = []
            
            self.draw_board()
    
    def make_move(self, move):
        """Execute a move on the board."""
        san_move = self.board.san(move)
        self.board.push(move)
        self.board_history.append(self.board.copy())
        self.move_history.append(san_move)
        
        self.selected_square = None
        self.legal_moves_from_selected = []
        
        self.update_display()
        self.update_move_history()
        
        if self.ai_loaded:
            self.root.after(100, self.update_ai_analysis)
    
    def update_display(self):
        """Update status and check game state."""
        self.draw_board()
        
        # Update status
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_label.config(text=f"Checkmate! {winner} wins!")
            messagebox.showinfo("Game Over", f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate!")
            messagebox.showinfo("Game Over", "Stalemate!")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Draw - Insufficient material")
            messagebox.showinfo("Game Over", "Draw by insufficient material!")
        elif self.board.is_check():
            side = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"{side} in check!")
        else:
            side = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"{side} to move")
    
    def update_ai_analysis(self):
        """Update AI predictions and evaluation."""
        if not self.ai_loaded or self.board.is_game_over():
            return
        
        try:
            # Get top moves
            top_moves = self.engine.predict_move(self.board_history, top_k=5)
            
            self.moves_text.config(state=tk.NORMAL)
            self.moves_text.delete(1.0, tk.END)
            self.moves_text.insert(tk.END, "Top 5 moves:\n\n")
            
            for i, (move, prob) in enumerate(top_moves, 1):
                san = self.board.san(move)
                self.moves_text.insert(tk.END, f"{i}. {san:8} ({prob:.2%})\n")
            
            self.moves_text.config(state=tk.DISABLED)
            
            # Get position evaluation
            value = self.engine.predict_value(self.board_history)
            
            side = "White" if self.board.turn == chess.WHITE else "Black"
            self.eval_label.config(
                text=f"{side} perspective:\n"
                     f"Win: {value['win']:.1%} | Draw: {value['draw']:.1%} | Loss: {value['loss']:.1%}"
            )
            
            # Update progress bars
            self.win_bar['value'] = value['win'] * 100
            self.win_label.config(text=f"{value['win']:.1%}")
            
            self.draw_bar['value'] = value['draw'] * 100
            self.draw_label.config(text=f"{value['draw']:.1%}")
            
            self.loss_bar['value'] = value['loss'] * 100
            self.loss_label.config(text=f"{value['loss']:.1%}")
            
        except Exception as e:
            print(f"Error updating AI analysis: {e}")
    
    def update_move_history(self):
        """Update the move history display."""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        for i in range(0, len(self.move_history), 2):
            move_num = i // 2 + 1
            white_move = self.move_history[i]
            black_move = self.move_history[i + 1] if i + 1 < len(self.move_history) else ""
            self.history_text.insert(tk.END, f"{move_num}. {white_move:8} {black_move}\n")
        
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)
    
    def make_ai_move(self):
        """Make the AI's suggested best move."""
        if not self.ai_loaded:
            messagebox.showwarning("AI Not Loaded", 
                                 "AI model is not loaded. Cannot make AI move.")
            return
        
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", "The game is already over!")
            return
        
        try:
            best_move = self.engine.get_best_move(self.board_history)
            if best_move:
                self.make_move(best_move)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get AI move: {e}")
    
    def undo_move(self):
        """Undo the last move."""
        if len(self.board_history) <= 1:
            messagebox.showinfo("Cannot Undo", "No moves to undo!")
            return
        
        self.board_history.pop()
        self.board = self.board_history[-1].copy()
        self.move_history.pop()
        
        self.selected_square = None
        self.legal_moves_from_selected = []
        
        self.update_display()
        self.update_move_history()
        
        if self.ai_loaded:
            self.root.after(100, self.update_ai_analysis)
    
    def new_game(self):
        """Start a new game."""
        if messagebox.askyesno("New Game", "Start a new game?"):
            self.board = chess.Board()
            self.board_history = [self.board.copy()]
            self.selected_square = None
            self.legal_moves_from_selected = []
            self.move_history = []
            
            self.update_display()
            self.update_move_history()
            
            if self.ai_loaded:
                self.root.after(100, self.update_ai_analysis)
    
    def flip_board(self):
        """Flip the board orientation."""
        self.flipped = not self.flipped
        self.draw_board()


def main():
    """Launch the ChessFormer GUI."""
    root = tk.Tk()
    
    # Default checkpoint path
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoint", "best_chessformer.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        result = messagebox.askyesnocancel(
            "Model Not Found",
            f"Checkpoint not found at:\n{checkpoint_path}\n\n"
            "Do you want to continue without AI features?"
        )
        if result is None:  # Cancel
            return
        elif result is False:  # No
            return
        # Yes - continue without AI
        checkpoint_path = None
    
    app = ChessGUI(root, checkpoint_path)
    root.mainloop()


if __name__ == "__main__":
    main()
