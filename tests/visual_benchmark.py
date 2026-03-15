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

def print_game_state(simulator, step_num, x, rot):
    """
    Prints the board and game information in the style of orchestrator/engine.py
    """
    board = simulator.board
    current_piece = simulator.getCurrentPiece()
    score = simulator.total_score
    is_over = simulator.is_game_over
    
    chars = {
        p.Cell.Red: "R",
        p.Cell.Green: "G",
        p.Cell.Blue: "B",
        p.Cell.Yellow: "Y",
        p.Cell.Ojama: "O",
        p.Cell.Empty: "."
    }
    
    print(f"Step {step_num}: Moved to x={x}, rot={rot}")
    print(f"Total Score: {score}")
    
    # Rows 0 to 12 are visible
    for y in range(12, -1, -1):
        line = f"{y:2d} | "
        for x_coord in range(6):
            line += chars.get(board.get(x_coord, y), "?")
        print(line)
    print("    " + "-" * 6)
    print("      012345")
    
    if is_over:
        print("!!! GAME OVER !!!")
    else:
        print(f"Next Piece: Axis={chars.get(current_piece.axis)}, Sub={chars.get(current_piece.sub)}")
    print()

def run_visual_benchmark(seed=1):
    print(f"Starting Visual Benchmark (Seed={seed})...")
    print("-" * 40)
    
    sim = p.Simulator(seed)
    step_num = 0
    
    # Move pattern from tests/benchmark.py
    # 6 moves at col 5, 6 at col 4, 6 at col 3, then col 2 until game over.
    move_plan = []
    for _ in range(6): move_plan.append(5)
    for _ in range(6): move_plan.append(4)
    for _ in range(6): move_plan.append(3)
    
    # Phase 1-3
    for x in move_plan:
        if sim.is_game_over: break
        step_num += 1
        sim.step(x, p.Rotation.Up)
        print_game_state(sim, step_num, x, p.Rotation.Up)
        # Use simple input to pause if needed, but for now just print all
        # time.sleep(0.1)

    # Phase 4: col 2 until game over
    while not sim.is_game_over:
        step_num += 1
        x = 2
        sim.step(x, p.Rotation.Up)
        print_game_state(sim, step_num, x, p.Rotation.Up)
        if step_num > 100: # Safety break
            print("Safety break: Step count exceeded 100")
            break

    print(f"Benchmark finished in {step_num} steps.")

if __name__ == "__main__":
    run_visual_benchmark(seed=1)
