import sys
from pathlib import Path

# Add the native library path
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
for d in [DIST_DIR, RELEASE_DIR]:
    if d.exists():
        sys.path.insert(0, str(d))
        break

import puyotan_native as p

sim = p.Simulator(1)

total_steps = 0
for col in [5, 4, 3]:
    for _ in range(6):
        if sim.is_game_over: break
        sim.step(col, p.Rotation.Up)
        total_steps += 1

print(f"Steps after phase 1-3: {total_steps}")
while not sim.is_game_over:
    sim.step(2, p.Rotation.Up)
    total_steps += 1

print(f"Final stops: {total_steps}")
