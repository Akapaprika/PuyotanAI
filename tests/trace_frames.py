import os
import sys

# Add dist to path
sys.path.append(os.path.join(os.getcwd(), 'native', 'dist'))

import puyotan_native as p

def print_board(board):
    for y in range(12, -1, -1):
        line = f"{y:2d} | "
        for x in range(6):
            cell = board.get(x, y)
            if cell == p.Cell.Red: line += "R"
            elif cell == p.Cell.Green: line += "G"
            elif cell == p.Cell.Blue: line += "B"
            elif cell == p.Cell.Yellow: line += "Y"
            elif cell == p.Cell.Ojama: line += "O"
            else: line += "."
        print(line)

match = p.PuyotanMatch(seed=1)
match.start() # Frame 1

print(f"--- Frame {match.frame} ---") # 1
print(f"can step: {match.canStepNextFrame()}")
# Set action for frame 1
match.setAction(0, p.Action(p.ActionType.PUT, 0, p.Rotation.Up))
match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
match.stepNextFrame() 
# Now at start of frame 2

print(f"--- Frame {match.frame} ---") # 2
print_board(match.getPlayer(0).field)
print(f"Action histories P1: {match.getPlayer(0).action_histories}")

# Set action for frame 2 (idle waiting)
match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
match.stepNextFrame()
# Now at start of frame 3

print(f"--- Frame {match.frame} ---") # 3
print_board(match.getPlayer(0).field)
print(f"Action histories P1: {match.getPlayer(0).action_histories}")

# Can we act?
is_idle = match.setAction(0, p.Action(p.ActionType.PUT, 1, p.Rotation.Up))
print(f"P1 can set action at F3? {is_idle}")
match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
match.stepNextFrame()
# Now at start of frame 4

print(f"--- Frame {match.frame} ---") # 4
print_board(match.getPlayer(0).field)
print(f"Action histories P1: {match.getPlayer(0).action_histories}")

match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
match.stepNextFrame()
# Now at start of frame 5

print(f"--- Frame {match.frame} ---") # 5
print_board(match.getPlayer(0).field)
print(f"Action histories P1: {match.getPlayer(0).action_histories}")

