import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import puyotan_env
import numpy as np

def test_single_env():
    print("Testing PuyotanEnv (Single)...")
    env = puyotan_env.PuyotanEnv(seed=1)
    obs, info = env.reset()
    print(f"Initial Obs Shape: {obs.shape}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}")
        if terminated:
            print("Game Over!")
            break
    print("Single environment test passed.\n")

def test_vector_env():
    num_envs = 10
    print(f"Testing PuyotanVectorEnv (N={num_envs})...")
    env = puyotan_env.PuyotanVectorEnv(num_envs=num_envs, base_seed=42)
    obs, info = env.reset()
    print(f"Initial Vector Obs Shape: {obs.shape}")
    
    for i in range(10):
        actions = [env.action_space.sample() for _ in range(num_envs)]
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"Vector Step {i+1}: Mean Reward={np.mean(rewards):.2f}, Done Count={np.sum(terminated)}")
    print("Vector environment test passed.")

if __name__ == "__main__":
    test_single_env()
    test_vector_env()
