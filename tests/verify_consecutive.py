import os
import sys

# Add dist to path
sys.path.append(os.path.join(os.getcwd(), 'native', 'dist'))

import puyotan_native as p

def verify_consecutive_puts():
    print("--- Consecutive PUTs Verification ---")
    match = p.PuyotanMatch(seed=1)
    match.start() # frame 1
    
    # We want to place piece 1, and immediately piece 2.
    # Player 1 actions
    # F1: issue PUT
    match.setAction(0, p.Action(p.ActionType.PUT, 0, p.Rotation.Up))
    match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
    match.stepNextFrame() # advances to frame 2
    
    # F2: wait
    match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
    match.stepNextFrame() # advances to frame 3. PUT should execute here.
    
    print(f"Start of Frame {match.frame}:")
    p1 = match.getPlayer(0)
    print(f"  P1 occupied: {not p1.field.getOccupied().empty()}")
    
    # F3: Player 1 should be ready for next PUT
    is_idle_f3 = match.setAction(0, p.Action(p.ActionType.PUT, 1, p.Rotation.Up))
    print(f"  P1 ready for next action? {is_idle_f3}")
    
    match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
    match.stepNextFrame() # advances to frame 4
    
    # F4: wait
    match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
    match.stepNextFrame() # advances to frame 5. 2nd PUT should execute here.
    
    # We can check simple occupation manually by masking lo and hi
    occ_lo = p1.field.getOccupied().lo
    occ_hi = p1.field.getOccupied().hi
    print(f"  P1 occupied lo: {occ_lo:016x}, hi: {occ_hi:016x}")
    print(f"  P1 occupied 2 pieces? {(occ_lo != 0 or occ_hi != 0)}")
    is_idle_f5 = match.setAction(0, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
    print(f"  P1 ready for next action? {is_idle_f5}")

if __name__ == "__main__":
    verify_consecutive_puts()
