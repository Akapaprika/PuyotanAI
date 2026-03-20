import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def test_game():
    print("Testing a single match with benchmark moves...")
    match = p.PuyotanMatch(1)
    match.start()
    
    move_plan = [5]*6 + [4]*6 + [3]*6
    p1_move = 0
    p2_move = 0
    
    start_time = time.time()
    while match.status == p.MatchStatus.PLAYING:
        if match.frame > 5000:
            print("FAILED: Too many frames")
            break
            
        p1_type = match.getPlayer(0).action_histories[match.frame & 255].action.type
        p2_type = match.getPlayer(1).action_histories[match.frame & 255].action.type
        
        if p1_type == p.ActionType.NONE:
            col = move_plan[p1_move] if p1_move < len(move_plan) else 2
            if match.setAction(0, p.Action(p.ActionType.PUT, col, p.Rotation.Up)):
                p1_move += 1
        
        if p2_type == p.ActionType.NONE:
            col = move_plan[p2_move] if p2_move < len(move_plan) else 2
            if match.setAction(1, p.Action(p.ActionType.PUT, col, p.Rotation.Up)):
                p2_move += 1
                
        if match.canStepNextFrame():
            match.stepNextFrame()
        else:
            print(f"STALL at frame {match.frame}. P1 type: {p1_type}, P2 type: {p2_type}")
            # This would be an infinite loop if we don't break
            break
            
        if time.time() - start_time > 2:
            print(f"TIMEOUT at frame {match.frame}")
            break

    print(f"Finished at frame {match.frame}. Status: {match.status_text}")
    print(f"P1 score: {match.getPlayer(0).score}, P2 score: {match.getPlayer(1).score}")

if __name__ == "__main__":
    test_game()
