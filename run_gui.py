#!/usr/bin/env python3
"""
ChessFormer Launcher

Quick launcher script for the ChessFormer GUI application.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the GUI
from gui.chess_gui import main

if __name__ == "__main__":
    main()
