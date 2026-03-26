import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def run_native_benchmark(num_games=10000, seed=1):
    print(f"[ PuyotanMatch NATIVE C++ Benchmark ] {num_games} games, seed={seed}")
    
    start = time.perf_counter()
    total_frames = p.PuyotanMatch.runBatch(num_games, seed)
    elapsed = time.perf_counter() - start

    print(f"  Games:           {num_games}")
    print(f"  Total Frames:    {total_frames}")
    print(f"  Avg Frames/Game: {total_frames / num_games:.1f}")
    print(f"  Total Time:      {elapsed:.4f} s")
    print(f"  Time/Game:       {elapsed / num_games * 1000000:.4f} μs")
    print(f"  Throughput:      {total_frames / elapsed:,.0f} frames/sec")
    print(f"  Throughput:      {num_games / elapsed:,.0f} games/sec")

if __name__ == "__main__":
    run_native_benchmark(1000000)
