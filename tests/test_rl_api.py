import sys
from pathlib import Path
BASE_DIR = Path(r'g:/マイドライブ/資料/プログラミング/プロジェクト/PuyotanAI')
for d in [BASE_DIR/'native'/'dist', BASE_DIR/'native'/'build_Release'/'Release']:
    if d.exists(): sys.path.insert(0, str(d)); break
import puyotan_native as p

# Test 1: step() returns delta score
sim = p.Simulator(seed=1)
delta = sim.step(2, p.Rotation.Up)
print(f"Test 1 - step() delta={delta}, total={sim.total_score}, type={type(delta).__name__}")
assert isinstance(delta, int)
print("  PASSED")

# Test 2: to_obs_flat() - check it returns a pybind11 array
obs = sim.board.to_obs_flat()
obs_type = type(obs).__name__
print(f"Test 2 - to_obs_flat type={obs_type}")
print(f"  shape={obs.shape}, dtype={obs.dtype}")
assert obs.shape == (5, 6, 13)
print("  PASSED")

# Test 3: clone() - deep copy isolation
match = p.PuyotanMatch(seed=1)
match.start()
match.setAction(0, p.Action(p.ActionType.PUT, 2, p.Rotation.Up))
match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
match.stepNextFrame()
frame_snap = match.frame
cloned = match.clone()
match.setAction(0, p.Action(p.ActionType.PUT, 3, p.Rotation.Up))
match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
match.stepNextFrame()
print(f"Test 3 - original frame={match.frame}, clone frame={cloned.frame} (expected {frame_snap})")
assert cloned.frame == frame_snap
print("  PASSED")

print("\nAll tests PASSED!")
