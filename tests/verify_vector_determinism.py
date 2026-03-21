import sys
from pathlib import Path

# Add native binary paths
BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def verify_vector(num_games=50000):
    vm = p.PuyotanVectorMatch(num_games, 0)
    for i in range(num_games):
        vm.get_match(i).start()

    total_frames = 0
    unfinished = set(range(num_games))
    moves_made = [[0, 0] for _ in range(num_games)]
    
    while unfinished:
        masks = vm.step_until_decision()
        match_indices = []
        player_ids = []
        actions = []
        
        to_remove = []
        for i in unfinished:
            mask = masks[i]
            if mask == 0:
                total_frames += vm.get_match(i).frame
                to_remove.append(i)
                continue
            
            for pid in range(2):
                if mask & (1 << pid):
                    m = vm.get_match(i)
                    count = moves_made[i][pid]
                    if count < 6: col = 5
                    elif count < 12: col = 4
                    elif count < 18: col = 3
                    else: col = 2
                    
                    match_indices.append(i)
                    player_ids.append(pid)
                    actions.append(p.Action(p.ActionType.PUT, col, p.Rotation.Up))
                    moves_made[i][pid] += 1
        
        for i in to_remove:
            unfinished.remove(i)
        if match_indices:
            vm.set_actions(match_indices, player_ids, actions)

    return total_frames

if __name__ == "__main__":
    frames = verify_vector(50000)
    print(f"Vectorized Total Frames: {frames}")
    if frames == 3061533:
        print("SUCCESS: Matches Gold Standard!")
    else:
        print(f"FAILURE: Expected 3061533, got {frames}")
