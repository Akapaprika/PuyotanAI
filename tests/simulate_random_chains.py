import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR/"native"/"dist", BASE_DIR/"native"/"build_Release"/"Release"]:
    if d.exists(): sys.path.insert(0, str(d)); break

import puyotan_native as p

def simulate():
    NUM_ENVS = 256
    STEPS = 128
    
    vm = p.PuyotanVectorMatch(NUM_ENVS, 42)
    
    all_chains = np.zeros((STEPS, NUM_ENVS), dtype=np.int32)
    obs_buffer = np.zeros((NUM_ENVS, 2, 5, 6, 14), dtype=np.uint8)
    import torch
    from training.model import PuyotanPolicy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PuyotanPolicy(hidden_dim=128).to(device)
    model.load_state_dict(torch.load("models/puyotan_latest.pt", map_location=device))
    model.eval()

    for t in range(STEPS):
        obs_tensor = torch.from_numpy(obs_buffer).float().to(device)
        with torch.no_grad():
            actions, _, _, _ = model.get_action_and_value(obs_tensor)
        actions = actions.cpu().numpy().astype(np.int32)
        obs, rewards, term, chains = vm.step(actions, None, obs_buffer)
        all_chains[t] = chains
    
    # Calculate Mean Chain Per Game (summing all chains in a game)
    chains_per_game = np.sum(all_chains, axis=0)
    print(f"Mean Sum of Chains per Game (128 drops): {np.mean(chains_per_game):.3f}")
    
    # Calculate Max Chain achieved in the entire batch
    print(f"Absolute Max Chain achieved by any game: {np.max(all_chains)}")
    
    # Max Chain per game
    max_per_game = np.max(all_chains, axis=0)
    print(f"Mean of Max Chain per Game: {np.mean(max_per_game):.3f}")
    
    # Total chains > 0
    non_zeros = all_chains[all_chains > 0]
    print(f"Average length of a chain when a chain occurs: {np.mean(non_zeros) if len(non_zeros) > 0 else 0:.3f}")

if __name__ == "__main__":
    simulate()
