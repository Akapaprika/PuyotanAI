import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release", BASE_DIR/"native"/"build_Debug"/"Debug"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

with open("debug_tracelog.txt", "w") as f:
    f.write("Starting...\n")
    match = p.PuyotanMatch(1)
    match.start()

    move_plan = [5]*6 + [4]*6 + [3]*6
    p1_move, p2_move = 0, 0
    
    for loop in range(1, 1000): # max 1000 loops
        if match.status != p.MatchStatus.PLAYING:
            f.write(f"Game over at Status: {match.status}. Exiting.\n")
            break

        f.write(f"Loop {loop}, Frame {match.frame}:\n")
        f.flush()

        p1_type = match.getPlayer(0).action_histories[match.frame & 255].action.type
        p2_type = match.getPlayer(1).action_histories[match.frame & 255].action.type

        if p1_type == p.ActionType.NONE:
            c = move_plan[p1_move] if p1_move < len(move_plan) else 2
            match.setAction(0, p.Action(p.ActionType.PUT, c, p.Rotation.Up))
            p1_move += 1
            f.write(f"  P1 set PUT col={c}\n")

        if p2_type == p.ActionType.NONE:
            c = move_plan[p2_move] if p2_move < len(move_plan) else 2
            match.setAction(1, p.Action(p.ActionType.PUT, c, p.Rotation.Up))
            p2_move += 1
            f.write(f"  P2 set PUT col={c}\n")
            
        f.flush()
            
        can = match.canStepNextFrame()
        f.write(f"  canStepNextFrame() = {can}\n")
        f.flush()
        
        if can:
            match.stepNextFrame()
            f.write(f"  stepNextFrame() done\n")
        else:
            f.write("  STALL! Neither was set?\n")
            break
        
        f.write("---\n")
        f.flush()

    f.write("Test completed.\n")
