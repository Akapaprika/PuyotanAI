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
    mask = match.step_until_decision()
    print(f"Step until decision returned mask: {mask} at frame {match.frame}")
    assert mask == 3, "Both players should need an action at start"
    
    # Test to_obs_flat
    p1 = match.getPlayer(0)
    obs = p1.to_obs_flat()
    print(f"Observation shape: {obs.shape}")
    assert obs.shape == (5, 6, 13)
    assert np.all(obs == 0), "Board should be empty at start"

    # Set actions and step
    match.setAction(0, p.Action(p.ActionType.PUT, 2, p.Rotation.Up))
    match.setAction(1, p.Action(p.ActionType.PUT, 3, p.Rotation.Up))
    
    mask = match.step_until_decision()
    print(f"Next decision at frame {match.frame}, mask: {mask}")
    
    # Verify observation after a move
    obs = match.getPlayer(0).to_obs_flat()
    print(f"Sum of obs after move: {np.sum(obs)}")
    assert np.sum(obs) == 2, "P1 should have placed 2 puyos"

    # Benchmarking step_until_decision efficiency
    print("\n--- Benchmarking step_until_decision ---")
    num_games = 50000
    start = time.perf_counter()
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
