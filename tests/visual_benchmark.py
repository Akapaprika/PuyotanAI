import sys
import os
from pathlib import Path
import time

# Add the native library path
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"

for d in [DIST_DIR, RELEASE_DIR]:
    if d.exists():
        sys.path.insert(0, str(d))
        break

try:
    import puyotan_native as p
except ImportError as e:
    print(f"Error: Could not import puyotan_native. Make sure to build the project first.\n{e}")
    sys.exit(1)

import argparse

def print_game_state(match, player_id=0):
    """
    Prints the board and game information including frame count.
    """
    player = match.getPlayer(player_id)
    board = player.field
    score = player.score
    status = match.status_text
    frame = match.frame
    
    chars = {
        p.Cell.Red: "R",
        p.Cell.Green: "G",
        p.Cell.Blue: "B",
        p.Cell.Yellow: "Y",
        p.Cell.Ojama: "O",
        p.Cell.Empty: "."
    }
    
    print(f"--- Frame: {frame} | Status: {status} ---")
    print(f"Player {player_id + 1} Score: {score}")
    
    # Rows 0 to 12 are visible
    for y in range(12, -1, -1):
        line = f"{y:2d} | "
        for x_coord in range(6):
            line += chars.get(board.get(x_coord, y), "?")
        print(line)
    print("    " + "-" * 6)
    print("      012345")
    print()

def run_visual_benchmark(seed=1, speed=0.05):
    print(f"Starting Visual Benchmark with Frame Info (Seed={seed})...")
    print("-" * 40)
    
    match = p.PuyotanMatch(seed)
    match.start()
    
    # Move plan: col 5x6, col 4x6, col 3x6, then col 2 until death
    # This matches the move sequence logic in benchmark.py
    move_plan = []
    move_plan += [5] * 6
    move_plan += [4] * 6
    move_plan += [3] * 6
    
    move_idx = 0
    
    while match.status == p.MatchStatus.PLAYING:
        p1 = match.getPlayer(0)
        p2 = match.getPlayer(1)

        # 1. Fill P1 action if needed
        if p1.action_histories[match.frame & 255].action.type == p.ActionType.NONE:
            x = move_plan[move_idx] if move_idx < len(move_plan) else 2
            if match.setAction(0, p.Action(p.ActionType.PUT, x, p.Rotation.Up)):
                move_idx += 1

        # 2. Fill P2 action if needed
        if p2.action_histories[match.frame & 255].action.type == p.ActionType.NONE:
            match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))

        # 3. Advance frame when both inputs are ready
        if match.canStepNextFrame():
            match.stepNextFrame()
            if speed > 0:
                print_game_state(match, 0)
                time.sleep(speed)

        if match.frame > 500: # Safety break
            print("Safety break: Frame count exceeded 500")
            break

    print(f"Benchmark finished at frame {match.frame}.")
    print(f"Final Status: {match.status_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--speed", type=float, default=0.05)
    args = parser.parse_args()
    
    run_visual_benchmark(seed=args.seed, speed=args.speed)
