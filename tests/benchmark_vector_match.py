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
    matches = [p.PuyotanMatch(i + 1) for i in range(num_games)]
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
                mask = m.stepUntilDecision()
                if mask == 0: 
                    total_frames += m.frame - 1
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
                total_frames += m.frame - 1 # Already finished somehow?
        unfinished = still_alive
        
    elapsed = time.perf_counter() - start_time
    return total_frames, elapsed

def benchmark_vectorized(num_games=1000):
    vm = p.PuyotanVectorMatch(num_games, 1)
    for i in range(num_games):
        vm.getMatch(i).start()

    total_frames = 0
    start_time = time.perf_counter()
    
    unfinished = set(range(num_games))
    moves_made = [[0, 0] for _ in range(num_games)]
    
    # Tracking observation overhead separately for info
    obs_time = 0.0
    obs_calls = 0

    while unfinished:
        # REALISTIC: Retrieve observation EVERY step (simulating policy input)
        t0 = time.perf_counter()
        obs = vm.getObservationsAll()
        # Simulate accessing the data (e.g. data validation or dummy sum)
        _ = obs.shape[0] 
        obs_time += (time.perf_counter() - t0)
        obs_calls += 1

        masks = vm.stepUntilDecision()
        
        match_indices = []
        player_ids = []
        actions = []
        
        to_remove = []
        for i in unfinished:
            mask = masks[i]
            if mask == 0:
                total_frames += vm.getMatch(i).frame - 1
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
            vm.setActions(match_indices, player_ids, actions)

    elapsed = time.perf_counter() - start_time
    
    print(f"Realistic Vectorized Stats:")
    print(f"  Obs Retrieval Total: {obs_time:.4f}s ({obs_calls} calls)")
    print(f"  Avg Obs Time: {obs_time/obs_calls if obs_calls > 0 else 0:.6f}s")
    
    return total_frames, elapsed

if __name__ == "__main__":
    N = 10000  # Realistic N for strategy-based benchmark
    EXPECTED_FRAMES = 602415
    
    print(f"--- Sequential Benchmark (N={N}) ---")
    f_seq, t_seq = benchmark_sequential(N)
    match_status_seq = "OK" if f_seq == EXPECTED_FRAMES else "FAIL"
    print(f"Total Frames: {f_seq} (Expected: {EXPECTED_FRAMES}) [{match_status_seq}]")
    print(f"Time: {t_seq:.4f}s")
    print(f"Throughput: {f_seq/t_seq:.2f} frames/sec")

    print(f"\n--- Vectorized Benchmark (N={N}) ---")
    f_vec, t_vec = benchmark_vectorized(N)
    match_status_vec = "OK" if f_vec == EXPECTED_FRAMES else "FAIL"
    print(f"Total Frames: {f_vec} (Expected: {EXPECTED_FRAMES}) [{match_status_vec}]")
    print(f"Time: {t_vec:.4f}s")
    print(f"Throughput: {f_vec/t_vec:.2f} frames/sec")
    
    print(f"\nSpeedup: {(f_vec/t_vec)/(f_seq/t_seq):.2f}x")
    assert f_seq == EXPECTED_FRAMES and f_vec == EXPECTED_FRAMES, "Behavior changed! Total frames mismatch."
