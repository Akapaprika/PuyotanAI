import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import env as puyotan_env
import numpy as np

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
    test_vector_env()
