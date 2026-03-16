import os
import sys

# Add dist to path
sys.path.append(os.path.join(os.getcwd(), 'native', 'dist'))

import puyotan_native as puyotan

def test_match():
    match = puyotan.PuyotanMatch(seed=123)
    print(f"Initial state: {match.status_text}, frame: {match.frame}")
    
    match.start()
    print(f"After start: {match.status_text}, frame: {match.frame}")
    
    # 1P and 2P actions
    # Action(type, x, rotation)
    match.setAction(0, puyotan.Action(puyotan.ActionType.PUT, 2, puyotan.Rotation.Up))
    match.setAction(1, puyotan.Action(puyotan.ActionType.PUT, 3, puyotan.Rotation.Up))
    
    print(f"Can step next frame? {match.canStepNextFrame()}")
    
    # Step until execution (2 frames for PUT)
    for _ in range(5):
        if match.canStepNextFrame():
            match.stepNextFrame()
            print(f"Frame {match.frame}: {match.status_text}")
            p1 = match.getPlayer(0)
            p2 = match.getPlayer(1)
            # Check puyos on board after execution
            # Note: Piece placement happens at frame execution (remaining_frame == 0)
            # Frame 1: remaining_frame=1
            # Frame 2: remaining_frame=0 -> Execute
            if match.frame > 2:
                # Check column 2 for P1 and column 3 for P2
                # (Highly simplified, just checking occupancy)
                occ1 = not p1.field.getOccupied().empty()
                occ2 = not p2.field.getOccupied().empty()
                print(f"  P1 occupied: {occ1}, P2 occupied: {occ2}")
        else:
            # Player needs to set next action
            match.setAction(0, puyotan.Action(puyotan.ActionType.PASS, 0, puyotan.Rotation.Up))
            match.setAction(1, puyotan.Action(puyotan.ActionType.PASS, 0, puyotan.Rotation.Up))

if __name__ == "__main__":
    test_match()
