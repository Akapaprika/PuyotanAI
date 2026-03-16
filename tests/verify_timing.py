import os
import sys

# Add dist to path
sys.path.append(os.path.join(os.getcwd(), 'native', 'dist'))

import puyotan_native as p

def verify_timing(title, board_setup_func):
    print(f"\n--- {title} ---")
    match = p.PuyotanMatch(seed=123)
    match.start()
    
    board_setup_func(match.getPlayer(0).field)
    
    # 1. First PUT
    match.setAction(0, p.Action(p.ActionType.PUT, 0, p.Rotation.Up))
    match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
    
    prev_frame = match.frame
    while True:
        if match.canStepNextFrame():
            match.stepNextFrame()
            # print(f"Frame {match.frame}: {match.status_text}")
            
            # Check if any player set a chain or something in the next frame
            # Or if it's the frame where CHAIN starts
            p1 = match.getPlayer(0)
            if match.frame in p1.action_histories:
                action_type = p1.action_histories[match.frame].action.type
                rem = p1.action_histories[match.frame].remaining_frame
                # print(f"  P1 Action at frame {match.frame}: {action_type}, rem={rem}")
        else:
            # Need next action for idle players
            acted = False
            for i in range(2):
                if match.frame not in match.getPlayer(i).action_histories:
                    match.setAction(i, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))
                    acted = True
            if not acted:
                break # Finished?
        
        if match.frame > 20: break

    # Find when CHAIN started and when it settled
    p1 = match.getPlayer(0)
    chain_start = 0
    chain_end = 0
    for f in range(1, match.frame + 1):
        if f in p1.action_histories:
            act = p1.action_histories[f].action.type
            if act == p.ActionType.CHAIN and chain_start == 0:
                chain_start = f
            if chain_start != 0 and act != p.ActionType.CHAIN and act != p.ActionType.CHAIN_FALL:
                if chain_end == 0:
                    chain_end = f
                    break
    
    if chain_end == 0: chain_end = match.frame
    
    print(f"Chain started at frame: {chain_start}")
    print(f"Chain settled at frame: {chain_end}")
    print(f"Duration: {chain_end - chain_start} frames")

def setup_quick(field):
    for x in range(4): field.set(x, 0, p.Cell.Red)

def setup_normal(field):
    for x in range(4): field.set(x, 1, p.Cell.Green)

if __name__ == "__main__":
    verify_timing("Quick Chain", setup_quick)
    verify_timing("Normal Chain", setup_normal)
