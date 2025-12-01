# ChessFormer 2.0 ♟️🤖

**ChessFormer 2.0** is a modern AI chess engine built using a custom Transformer-based neural network. Unlike traditional engines that rely on alpha-beta pruning and hand-crafted evaluation functions, ChessFormer learns to play chess by analyzing patterns in millions of grandmaster games.

This project includes the complete pipeline: from data processing and model training to a fully interactive GUI for playing against the AI.

---

## 🚀 Features

*   **Transformer Architecture**: Uses a specialized "Relative Transformer" designed to understand the spatial relationships on a chess board.
*   **Deep Learning**: Predicts the best move (Policy) and the game outcome (Value) directly from the board state.
*   **Interactive GUI**: A clean, Python-based graphical interface to play against the AI or analyze positions.
*   **Efficient Inference**: Optimized inference engine for quick move generation.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Chess_Former_2.0
```

### 2. Set Up Python Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Play Against the AI (GUI)
To start the graphical interface and play a game:

```bash
python gui/chess_gui.py
```
*   **Note**: The GUI will automatically look for a trained model in `checkpoint/best_chessformer.pt`. If none is found, it will run in "board-only" mode.

### Run Inference (CLI)
To test the model's predictions on a sample position without the GUI:

```bash
python inference/inference.py
```

### Train the Model
If you want to train the model yourself:

1.  Place your PGN file (e.g., `final_800k_elite.pgn`) in the `pgn-datasets/` directory.
2.  (Optional) Split large PGNs into chunks for easier processing:
    ```bash
    python pgn-datasets/split_pgn.py
    ```
3.  Run the training script:
    ```bash
    python chess_former.py
    ```

---

## 📂 Project Structure

Here is an overview of the key files in the repository:

### **Core Model**
*   **`model.py`**: Defines the `ChessFormer` architecture, including the policy (move) and value (win/loss) heads.
*   **`transformer_encoder.py`**: Implements a custom Transformer Encoder layer tailored for chess.
*   **`attention.py`**: Contains the **Relative Multi-Head Attention** mechanism, allowing the model to understand board geometry (diagonals, ranks, files).

### **Training & Data**
*   **`chess_former.py`**: The main training loop. Handles loading data, backpropagation, and saving checkpoints.
*   **`dataset.py`**: Responsible for parsing PGN files and converting chess board states into tensor representations for the model.
*   **`loader.py`**: Utilities for creating PyTorch DataLoaders for training and validation sets.
*   **`pgn-datasets/`**: Directory for storing raw PGN game files.
    *   **`split_pgn.py`**: A utility script to split massive PGN files into smaller, manageable chunks (e.g., 1000 games per file).

### **Application**
*   **`inference/inference.py`**: The bridge between the trained model and the user. It handles loading weights and generating move probabilities.
*   **`gui/chess_gui.py`**: The frontend application. A Tkinter-based chess board that visualizes the game, legal moves, and AI predictions.

### **Configuration**
*   **`requirements.txt`**: List of Python libraries required to run the project (e.g., `torch`, `python-chess`, `numpy`).
*   **`checkpoint/`**: Directory where trained model weights (`.pt` files) are saved.

---

## 🧠 How It Works

1.  **Input**: The model takes the last 8 board positions (history) to understand the game flow.
2.  **Processing**: A stack of Transformer layers analyzes the board. The "Relative Attention" mechanism helps the model "see" how pieces relate to each other spatially.
3.  **Output**:
    *   **Policy Head**: Predicts the probability of moving from every square to every other square.
    *   **Value Head**: Estimates the probability of White winning, Black winning, or a Draw.
    *   **Promotion Head**: Specifically handles pawn promotion decisions.

---

## 📝 To-Do
*   [ ] Add support for UCI protocol to use with standard chess GUIs (like Arena or Fritz).
*   [ ] Improve search depth using Monte Carlo Tree Search (MCTS).
