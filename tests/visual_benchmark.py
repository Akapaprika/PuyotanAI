import sys
import time
from pathlib import Path
import argparse

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
    print(f"Error: Could not import puyotan_native. build first.\n{e}")
    sys.exit(1)

def cell_to_char(cell):
    return {
        p.Cell.Red: "R",
        p.Cell.Green: "G",
        p.Cell.Blue: "B",
        p.Cell.Yellow: "Y",
        p.Cell.Ojama: "O",
        p.Cell.Empty: ".",
    }.get(cell, "?")

def format_tsumo(piece):
    return f"[{cell_to_char(piece.axis)}{cell_to_char(piece.sub)}]"

def print_game_state(match, moves_made):
    """Prints both players' boards side-by-side."""
    p1 = match.getPlayer(0)
    p2 = match.getPlayer(1)
    
    # Header
    print("\033[H\033[J", end="") # Clear screen (ANSI)
    print(f"=== Puyotan Visual Benchmark | Frame: {match.frame:4d} ===")
    
    # Next Queue
    tsumo = match.getTsumo()
    # P1/P2 have independent activeNextPos
    n1_p1 = tsumo.get(p1.active_next_pos)
    n2_p1 = tsumo.get(p1.active_next_pos + 1)
    n1_p2 = tsumo.get(p2.active_next_pos)
    n2_p2 = tsumo.get(p2.active_next_pos + 1)
    
    print(f"  P1 NEXT: {format_tsumo(n1_p1)} {format_tsumo(n2_p1)}        P2 NEXT: {format_tsumo(n1_p2)} {format_tsumo(n2_p2)}")
    print(f"  P1 Score: {p1.score:6d}  Ojama: {p1.active_ojama}({p1.non_active_ojama}) | P2 Score: {p2.score:6d}  Ojama: {p2.active_ojama}({p2.non_active_ojama})")
    print("-" * 64)

    # Boards side-by-side
    for y in range(12, -1, -1):
        line = f"{y:2d} | "
        # P1 Field
        for x in range(6):
            line += cell_to_char(p1.field.get(x, y))
        line += " |        "
        # P2 Field
        line += f"{y:2d} | "
        for x in range(6):
            line += cell_to_char(p2.field.get(x, y))
        line += " |"
        print(line)
    
    print("    " + "-" * 6 + "                 " + "-" * 6)
    print("      012345                 012345")
    print(f"  Moves: {moves_made[0]:3d}                   Moves: {moves_made[1]:3d}")
    print(f"  Chain: {p1.chain_count:3d}                   Chain: {p2.chain_count:3d}")
    print("-" * 64)

def run_visual_benchmark(seed=1, speed=0.05):
    match = p.PuyotanMatch(seed)
    match.start()
    
    moves_made = [0, 0]
    
    while match.status == p.MatchStatus.PLAYING:
        # 1. Advance simulation until someone needs a decision
        mask = match.stepUntilDecision()
        
        if mask == 0: # Match ended
            break
            
        # 2. Assign actions based on mask
        for pid in range(2):
            if mask & (1 << pid):
                # 5-4-3-2 strategy
                count = moves_made[pid]
                if count < 6: col = 5
                elif count < 12: col = 4
                elif count < 18: col = 3
                else: col = 2
                
                match.setAction(pid, p.Action(p.ActionType.PUT, col, p.Rotation.Up))
                moves_made[pid] += 1
        
        # 3. Advance one frame
        match.stepNextFrame()
        
        # 4. Display
        if speed > 0:
            print_game_state(match, moves_made)
            time.sleep(speed)
            
        if match.frame > 2000: # Safety
            break

    print_game_state(match, moves_made)
    print(f"Total Frames: {match.frame}")

    # --- Automated Verification ---
    p1 = match.getPlayer(0)
    p2 = match.getPlayer(1)

    assert match.frame == 57, f"Frame mismatch! Expected 57, got {match.frame}"
    assert p1.score == 280, f"P1 Score mismatch! Expected 280, got {p1.score}"
    assert p2.score == 280, f"P2 Score mismatch! Expected 280, got {p2.score}"

    expected_board = [
        "..Y...",
        "..Y...",
        "..B.G.",
        "..R.R.",
        "..GRGR",
        "..GBYY",
        "..BYBY",
        "..GYGG",
        "..RGYR",
        "..BGBY",
        "..RRGG",
        "..BRBY",
        "..YYRY"
    ]

    for y, exp_row in zip(range(12, -1, -1), expected_board):
        p1_row = "".join(cell_to_char(p1.field.get(x, y)) for x in range(6))
        p2_row = "".join(cell_to_char(p2.field.get(x, y)) for x in range(6))
        assert p1_row == exp_row, f"P1 Row {y} mismatch! Expected {exp_row}, got {p1_row}"
        assert p2_row == exp_row, f"P2 Row {y} mismatch! Expected {exp_row}, got {p2_row}"
        
    print("\n[OK] Behavioral specifications matched 100% exactly (Frames, Scores, and Final Board states).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--speed", type=float, default=0.05)
    args = parser.parse_args()
    
    # Hide cursor
    print("\033[?25l", end="")
    try:
        run_visual_benchmark(seed=args.seed, speed=args.speed)
    except KeyboardInterrupt:
        pass
    finally:
        # Show cursor
        print("\033[?25h")
