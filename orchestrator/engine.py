import sys
import os
from pathlib import Path

# Add the native library path
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"

if DIST_DIR.exists():
    sys.path.insert(0, str(DIST_DIR))
else:
    # Fallback for development if dist doesn't exist yet
    DEBUG_DIR = BASE_DIR / "native" / "build" / "Debug"
    if DEBUG_DIR.exists():
        sys.path.insert(0, str(DEBUG_DIR))

try:
    import puyotan_native as p
except ImportError as e:
    print(f"Error: Could not import puyotan_native. Make sure to build the project first.\n{e}")
    sys.exit(1)

class PuyotanEngine:
    """
    High-level Python wrapper for the Puyotan C++ engine.
    """
    def __init__(self, seed=0):
        self.simulator = p.Simulator(seed)

    def reset(self, seed=0):
        """Resets the game with a new seed."""
        self.simulator.reset(seed)

    def move(self, x, rotation):
        """Places a piece at column x with specified rotation."""
        if not self.simulator.isGameOver():
            self.simulator.step(x, rotation)
            return True
        return False

    def is_game_over(self):
        return self.simulator.isGameOver()

    def get_board(self):
        return self.simulator.getBoard()

    def get_current_piece(self):
        return self.simulator.getCurrentPiece()

    def get_tsumo_index(self):
        return self.simulator.getTsumoIndex()

    def get_total_score(self):
        return self.simulator.getTotalScore()

    def print_board(self):
        """Simple ASCII representation of the board."""
        board = self.get_board()
        chars = {
            p.Cell.Red: "R",
            p.Cell.Green: "G",
            p.Cell.Blue: "B",
            p.Cell.Yellow: "Y",
            p.Cell.Ojama: "O",
            p.Cell.Empty: "."
        }
        
        # Rows 0 to 12 are visible
        for y in range(12, -1, -1):
            line = ""
            for x in range(6):
                line += chars.get(board.get(x, y), "?")
            print(line)
        print("-" * 6)

if __name__ == "__main__":
    # Simple demo
    print("Starting Puyotan Engine Demo...")
    engine = PuyotanEngine(seed=1)
    
    # Demo: Place a few pieces
    moves = [
        (3, p.Rotation.Up),
        (2, p.Rotation.Right),
        (4, p.Rotation.Left),
        (3, p.Rotation.Down),
    ]

    for x, rot in moves:
        piece = engine.get_current_piece()
        print(f"Step {engine.get_tsumo_index()}: Piece(axis={piece.axis}, sub={piece.sub}) -> x={x}, rot={rot}")
        engine.move(x, rot)
        engine.print_board()
        print(f"Total Score: {engine.get_total_score()}")
        print()

    if engine.is_game_over():
        print("Game Over!")
    else:
        print("Demo finished successfully.")
