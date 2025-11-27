# ChessFormer - Inference & GUI

This directory contains the inference engine and graphical user interface for the ChessFormer chess AI model.

## Files

- **`inference.py`** - Core inference engine that loads the trained model and provides move predictions
- **`chess_gui.py`** - Interactive GUI built with Tkinter for playing chess with AI assistance

## Quick Start

### Run the GUI

From the project root directory:

```bash
python run_gui.py
```

Or directly:

```bash
python gui/chess_gui.py
```

### Use Inference Programmatically

```python
from inference.inference import ChessFormerInference
import chess

# Load the model
engine = ChessFormerInference("checkpoint/best_chessformer.pt")

# Create a position
board = chess.Board()
board_history = [board.copy()]

# Get AI predictions
top_moves = engine.predict_move(board_history, top_k=5)
for move, prob in top_moves:
    print(f"{move.uci()}: {prob:.2%}")

# Get position evaluation
value = engine.predict_value(board_history)
print(f"Win: {value['win']:.1%}, Draw: {value['draw']:.1%}, Loss: {value['loss']:.1%}")

# Get best move
best_move = engine.get_best_move(board_history)
print(f"Best move: {best_move.uci()}")
```

## GUI Features

### Interactive Chess Board
- Click pieces to select them
- Click destination squares to make moves
- Visual highlights for selected pieces and legal moves
- Support for all chess rules including castling, en passant, and promotion

### AI Features
- **Move Suggestions**: Top 5 AI-predicted moves with probabilities
- **Position Evaluation**: Win/Draw/Loss percentages from current side's perspective
- **AI Move**: Let the AI make its best move
- **Real-time Analysis**: Updates after every move

### Controls
- **New Game**: Start a fresh game
- **Undo Move**: Take back the last move
- **AI Move**: Make the AI's top suggested move
- **Flip Board**: Rotate the board 180 degrees

### Display
- **Status Panel**: Shows whose turn it is and special states (check, checkmate, etc.)
- **Position Evaluation**: Visual progress bars showing win/draw/loss probabilities
- **AI Suggested Moves**: List of top moves with probabilities
- **Move History**: Complete game notation

## Requirements

```
torch
chess (python-chess)
tkinter (usually included with Python)
```

Install dependencies:

```bash
pip install torch python-chess
```

## Model Path

The GUI automatically looks for the trained model at:
```
checkpoint/best_chessformer.pt
```

If the model is not found, you'll be prompted to continue without AI features (manual play only).

## Architecture

### ChessFormerInference
The inference engine handles:
- Loading trained model weights
- Converting board positions to model input format
- Predicting legal moves with probabilities
- Evaluating position win/draw/loss chances
- Coordinate conversion between python-chess and model indices

### ChessGUI
The GUI provides:
- 640x640 pixel chess board with Unicode pieces
- Click-based move input
- Real-time AI analysis display
- Game state management
- Move history tracking

## Technical Details

### Coordinate Systems
The model uses a different square indexing than python-chess:
- **python-chess**: a1=0, h1=7, a8=56, h8=63 (rank 1 to 8, bottom to top)
- **Model**: a8=0, h8=7, a1=56, h1=63 (rank 8 to 1, top to bottom)

The inference engine handles all conversions automatically.

### Board History
The model uses 8 steps of board history for temporal awareness. The inference engine:
- Maintains the full board history
- Pads with initial position if history < 8 moves
- Uses last 8 positions if history > 8 moves

### Move Prediction
1. Model outputs 64×64 policy logits (from-square × to-square)
2. Softmax converts to probabilities
3. Top-k moves extracted
4. Filtered to only legal moves
5. Promotion moves handled with separate promotion head

## Troubleshooting

### "Model not found" error
- Ensure the checkpoint file exists at `checkpoint/best_chessformer.pt`
- Or modify the `checkpoint_path` in the script

### ImportError
- Make sure all dependencies are installed: `pip install torch python-chess`
- Ensure the project structure is intact

### Slow predictions
- Model runs on GPU if available (CUDA)
- First prediction may be slower (model initialization)
- Subsequent predictions should be fast

### GUI doesn't start
- Check tkinter is installed: `python -m tkinter`
- On Linux: `sudo apt-get install python3-tk`
- On macOS: tkinter comes with Python

## Enjoy!

Play chess against yourself with AI suggestions, or let the AI play its best moves. The ChessFormer model has been trained on hundreds of thousands of elite games!
