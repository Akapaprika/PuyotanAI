import sys
from pathlib import Path

# Add the native library path
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
for d in [DIST_DIR, RELEASE_DIR]:
    if d.exists():
        sys.path.insert(0, str(d))
        break

import puyotan_native as p

def print_board(board):
    for y in range(13, -1, -1):
        line = f"{y:2d} |"
        for x in range(6):
            cell = board.get(x, y)
            if cell == p.Cell.Red: line += "R"
            elif cell == p.Cell.Green: line += "G"
            elif cell == p.Cell.Blue: line += "B"
            elif cell == p.Cell.Yellow: line += "Y"
            elif cell == p.Cell.Ojama: line += "O"
            else: line += "."
        print(line)
    print("    ------")
    print("     012345")

sim = p.Simulator(1)

total_steps = 0
for col in [5, 4, 3]:
    for _ in range(6):
        sim.step(col, p.Rotation.Up)
        total_steps += 1

print(f"--- After phase 1-3 (Steps: {total_steps}) ---")
print_board(sim.board)

while not sim.is_game_over:
    sim.step(2, p.Rotation.Up)
    total_steps += 1
    print(f"--- After Step {total_steps} (Col 2 Piece {total_steps-18}) ---")
    print_board(sim.board)
    if total_steps > 30: break

print(f"Final steps: {total_steps}")
print(f"Game over at: {sim.is_game_over}")
