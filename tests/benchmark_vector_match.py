import sys
import time
from pathlib import Path
import numpy as np

# Add native binary paths
BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def benchmark_sequential(num_games=1000):
    matches = [p.PuyotanMatch(i) for i in range(num_games)]
    for m in matches: m.start()
    
    total_frames = 0
    start_time = time.perf_counter()
    
    unfinished = list(range(num_games))
    moves_made = [[0, 0] for _ in range(num_games)]
    
    while unfinished:
        still_alive = []
        for i in unfinished:
            m = matches[i]
            if m.status == p.MatchStatus.PLAYING:
                mask = m.step_until_decision()
                if mask == 0: 
                    total_frames += m.frame
                    continue
                
                for pid in range(2):
                    if mask & (1 << pid):
                        count = moves_made[i][pid]
                        if count < 6: col = 5
                        elif count < 12: col = 4
                        elif count < 18: col = 3
                        else: col = 2
                        m.setAction(pid, p.Action(p.ActionType.PUT, col, p.Rotation.Up))
                        moves_made[i][pid] += 1
                still_alive.append(i)
            else:
                total_frames += m.frame # Already finished somehow?
        unfinished = still_alive
        
    elapsed = time.perf_counter() - start_time
    return total_frames, elapsed

def benchmark_vectorized(num_games=1000):
    vm = p.PuyotanVectorMatch(num_games, 0)
    # No start() needed yet? Wait, PuyotanMatch needs start.
    # PuyotanVectorMatch constructor creates PuyotanMatch objects but they aren't started.
    for i in range(num_games):
        vm.get_match(i).start()

    total_frames = 0
    start_time = time.perf_counter()
    
    unfinished = set(range(num_games))
    moves_made = [[0, 0] for _ in range(num_games)]
    while unfinished:
        masks = vm.step_until_decision()
        
        match_indices = []
        player_ids = []
        actions = []
        
        to_remove = []
        for i in unfinished:
            mask = masks[i]
            if mask == 0:
                total_frames += vm.get_match(i).frame
                to_remove.append(i)
                continue
            
            for pid in range(2):
                if mask & (1 << pid):
                    count = moves_made[i][pid]
                    if count < 6: col = 5
                    elif count < 12: col = 4
                    elif count < 18: col = 3
                    else: col = 2
                    match_indices.append(i)
                    player_ids.append(pid)
                    actions.append(p.Action(p.ActionType.PUT, col, p.Rotation.Up))
                    moves_made[i][pid] += 1
        
        for i in to_remove:
            unfinished.remove(i)
            
        if match_indices:
            vm.set_actions(match_indices, player_ids, actions)

    # All frames are already summed in the while loop above.

    # Observation Benchmark
    start_obs = time.perf_counter()
    obs = vm.get_observations_all()
    t_obs = time.perf_counter() - start_obs
    print(f"Observation Generation (N={num_games}): {t_obs:.4f}s ({num_games/t_obs:.2f} obs/sec)")
    print(f"Observation Shape: {obs.shape}")

    elapsed = time.perf_counter() - start_time
    return total_frames, elapsed

if __name__ == "__main__":
    N = 10000  # Realistic N for strategy-based benchmark
    print(f"--- Sequential Benchmark (N={N}) ---")
    f_seq, t_seq = benchmark_sequential(N)
    print(f"Total Frames: {f_seq}")
    print(f"Time: {t_seq:.4f}s")
    print(f"Throughput: {f_seq/t_seq:.2f} frames/sec")

    print(f"\n--- Vectorized Benchmark (N={N}) ---")
    f_vec, t_vec = benchmark_vectorized(N)
    print(f"Total Frames: {f_vec}")
    print(f"Time: {t_vec:.4f}s")
    print(f"Throughput: {f_vec/t_vec:.2f} frames/sec")
    
    print(f"\nSpeedup: {(f_vec/t_vec)/(f_seq/t_seq):.2f}x")
