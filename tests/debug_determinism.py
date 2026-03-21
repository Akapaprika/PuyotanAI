import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def run_test(num_games=1000):
    total_frames = 0
    for i in range(num_games):
        m = p.PuyotanMatch(i)
        m.start()
        moves_made = [0, 0]
        while m.status == p.MatchStatus.PLAYING:
            mask = m.step_until_decision()
            if mask == 0: break
            
            for id in range(2):
                if mask & (1 << id):
                    # Policy: 6x col 5, 6x col 4, 6x col 3, then col 2
                    count = moves_made[id]
                    if count < 6: col = 5
                    elif count < 12: col = 4
                    elif count < 18: col = 3
                    else: col = 2
                    
                    if m.setAction(id, p.Action(p.ActionType.PUT, col, p.Rotation.Up)):
                        moves_made[id] += 1
        total_frames += m.frame
    return total_frames

if __name__ == "__main__":
    print("Run 1...")
    res1 = run_test(50000)
    print(f"Total Frames: {res1}")
    
    print("Run 2...")
    res2 = run_test(50000)
    print(f"Total Frames: {res2}")
    
    if res1 == res2:
        print("SUCCESS: Deterministic")
    else:
        print(f"FAILURE: Non-deterministic! Diff: {res2 - res1}")
