import sys
from pathlib import Path
import time
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def test_rl_api():
    print("--- Verifying RL Optimization APIs ---")
    match = p.PuyotanMatch(1)
    match.start()
    
    # Test step_until_decision
    print(f"Initial frame: {match.frame}")
    mask = match.stepUntilDecision()
    print(f"Step until decision returned mask: {mask} at frame {match.frame}")
    assert mask == 3, "Both players should need an action at start"

    # Set actions and step
    match.setAction(0, p.Action(p.ActionType.PUT, 2, p.Rotation.Up))
    match.setAction(1, p.Action(p.ActionType.PUT, 3, p.Rotation.Up))
    
    mask = match.stepUntilDecision()
    print(f"Next decision at frame {match.frame}, mask: {mask}")

    # Benchmarking stepUntilDecision efficiency
    print("\n--- Benchmarking stepUntilDecision ---")
    num_games = 50000
    start = time.perf_counter()
    total_frames = 0
    for i in range(num_games):
        m = p.PuyotanMatch(i)
        m.start()
        moves_made = [0, 0]
        while m.status == p.MatchStatus.PLAYING:
            mask = m.stepUntilDecision()
            if mask == 0: break
            for id in range(2):
                if mask & (1 << id):
                    count = moves_made[id]
                    if count < 6: col = 5
                    elif count < 12: col = 4
                    elif count < 18: col = 3
                    else: col = 2
                    if m.setAction(id, p.Action(p.ActionType.PUT, col, p.Rotation.Up)):
                        moves_made[id] += 1
        total_frames += m.frame

    elapsed = time.perf_counter() - start
    print(f"Results for {num_games} games:")
    print(f"  Total Frames: {total_frames}")
    print(f"  Total Time:   {elapsed:.4f} s")
    print(f"  Throughput:   {total_frames / elapsed:,.0f} frames/sec")

if __name__ == "__main__":
    test_rl_api()
