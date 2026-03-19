import time
import sys
from pathlib import Path

# Add the native library path
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
DEBUG_DIR = BASE_DIR / "native" / "build_Debug" / "Debug"

for d in [DIST_DIR, RELEASE_DIR, DEBUG_DIR]:
    if d.exists():
        sys.path.insert(0, str(d))
        break

try:
    import puyotan_native as p
except ImportError as e:
    print(f"Error: Could not import puyotan_native. Make sure to build the project first.\n{e}")
    sys.exit(1)

def run_benchmark(num_games=5000000, seed=1):
    """
    Measures throughput via two methods:
    1. Python-loop: step() called individually from Python (includes pybind11 boundary overhead)
    2. C++ batch: runBatch() runs all games in pure C++ without Python overhead (true engine speed)
    """
    print(f"Starting benchmark: {num_games} games, seed={seed}")
    print()

    # --- Method 1: Python-loop (step() per move) ---
    # print("[ Method 1: Python-loop ]")
    # total_steps = 0

    # start_time = time.perf_counter()
    # for i in range(num_games):
    #     sim = p.Simulator(seed)

    #     for _ in range(6):
    #         if sim.is_game_over: break
    #         sim.step(5, p.Rotation.Up)
    #         total_steps += 1

    #     for _ in range(6):
    #         if sim.is_game_over: break
    #         sim.step(4, p.Rotation.Up)
    #         total_steps += 1

    #     for _ in range(6):
    #         if sim.is_game_over: break
    #         sim.step(3, p.Rotation.Up)
    #         total_steps += 1

    #     while not sim.is_game_over:
    #         sim.step(2, p.Rotation.Up)
    #         total_steps += 1

    #     if (i + 1) % 100000 == 0:
    #         print(f"  Progress: {i + 1}/{num_games}")

    # elapsed = time.perf_counter() - start_time
    # print(f"  Total Steps:    {total_steps}")
    # print(f"  Total Time:     {elapsed:.4f} s")
    # print(f"  Time per Game:  {elapsed / num_games * 1000:.4f} ms")
    # print(f"  Throughput:     {total_steps / elapsed:,.0f} steps/sec")
    # print()

    # --- Method 2: C++ batch (runBatch — no Python overhead) ---
    print("[ Method 2: C++ batch (runBatch) ]")
    sim_batch = p.Simulator()

    start_time = time.perf_counter()
    batch_steps = sim_batch.runBatch(num_games, seed)
    elapsed = time.perf_counter() - start_time

    print(f"  Total Steps:    {batch_steps}")
    print(f"  Total Time:     {elapsed:.4f} s")
    print(f"  Time per Game:  {elapsed / num_games * 1000:.4f} ms")
    print(f"  Throughput:     {batch_steps / elapsed:,.0f} steps/sec")

if __name__ == "__main__":
    run_benchmark()
