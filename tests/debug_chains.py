import sys
from pathlib import Path
import time
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def print_board(match):
    obs = match.getObservationsAll().squeeze()
    print("Observer shape:", obs.shape)

def run():
    print("--- Debugging Chains ---")
    vm = p.PuyotanVectorMatch(1, 123)
    
    # Force a sequence of drops that MUST create a chain.
    # Player 0 actions: Let's just drop everything in column 0 (action 0: UP, col 0)
    # Eventually it will stack up and some colors will match.
    for i in range(500):
        actions = np.random.randint(0, 22, size=(1,), dtype=np.int32)
        obs, rewards, terminated, chains = vm.step(actions)
        
        print(f"Step {i+1}: Rewards={rewards[0]:.4f}, Terminated={terminated[0]}, Chain={chains[0]}")
        if chains[0] > 0:
            print(f">>> CHAIN DETECTED: {chains[0]}")
        
        if terminated[0]:
            print(f"Game terminated at step {i+1}")
            break

if __name__ == "__main__":
    run()
