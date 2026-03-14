import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"

sys.path.append(str(DIST_DIR))

import puyotan_native


def choose_move():
    state = puyotan_native.GameState()
    move = puyotan_native.chooseMove(state)
    return move.column

print(choose_move())
