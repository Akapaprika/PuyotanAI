import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
for d in [DIST_DIR, RELEASE_DIR]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

match = p.PuyotanMatch(1)
match.start()

print(f"Frame start: {match.frame}")

p1 = match.getPlayer(0)
p2 = match.getPlayer(1)

for step in range(3):
    print(f"\n--- Loop {step} Frame {match.frame} ---")
    p1 = match.getPlayer(0) # re-fetch to get latest copy!
    p2 = match.getPlayer(1)
    
    action1_type = p1.action_histories[match.frame & 255].action.type
    action2_type = p2.action_histories[match.frame & 255].action.type
    print(f"Before setAction: P1={action1_type}, P2={action2_type}")
    
    if action1_type == p.ActionType.NONE:
        res1 = match.setAction(0, p.Action(p.ActionType.PUT, 3, p.Rotation.Up))
        print(f"P1 setAction(PUT): {res1}")
    if action2_type == p.ActionType.NONE:
        res2 = match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
        print(f"P2 setAction(PASS): {res2}")
        
    p1 = match.getPlayer(0) 
    p2 = match.getPlayer(1)
    a1 = p1.action_histories[match.frame & 255]
    a2 = p2.action_histories[match.frame & 255]
    print(f"After setAction: P1={a1.action.type}(rem={a1.remaining_frame}), P2={a2.action.type}(rem={a2.remaining_frame})")
    
    can = match.canStepNextFrame()
    print(f"canStepNextFrame: {can}")
    if can:
        match.stepNextFrame()
        print(f"stepped! new frame: {match.frame}")

print("\nP1 score:", match.getPlayer(0).score)
