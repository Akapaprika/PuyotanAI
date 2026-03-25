import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path

# Add the project root and native library path
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"

for d in [DIST_DIR, RELEASE_DIR]:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))
        break

try:
    import puyotan_native as p
except ImportError as e:
    # This might happen during pip install if not built yet
    p = None

class ActionMapper:
    """Maps discrete action [0-21] to (col, rotation)."""
    # 0-5: Up, 6-10: Right, 11-16: Down, 17-21: Left
    ACTIONS = []
    
    @classmethod
    def initialize(cls):
        if cls.ACTIONS: return
        if p is None: return
        for r in [p.Rotation.Up]: # col 0-5
            for x in range(6): cls.ACTIONS.append((x, r))
        for r in [p.Rotation.Right]: # col 0-4
            for x in range(5): cls.ACTIONS.append((x, r))
        for r in [p.Rotation.Down]: # col 0-5
            for x in range(6): cls.ACTIONS.append((x, r))
        for r in [p.Rotation.Left]: # col 1-5
            for x in range(1, 6): cls.ACTIONS.append((x, r))

    @classmethod
    def get(cls, idx):
        cls.initialize()
        return cls.ACTIONS[idx]

class PuyotanVectorEnv:
    """
    Hig-performance Vectorized environment for Puyotan!
    Directly wraps PuyotanVectorMatch for massive parallel training.
    """
    def __init__(self, num_envs=1, base_seed=1):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.vm = p.PuyotanVectorMatch(num_envs, base_seed)
        
        # Ensure ActionMapper is ready
        ActionMapper.initialize()
        self.action_space = spaces.Discrete(len(ActionMapper.ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, 5, 6, 13), dtype=np.uint8
        )
        self._obs_buffer = np.zeros((num_envs, 2, 5, 6, 13), dtype=np.uint8)
        self._truncated_buffer = np.zeros(num_envs, dtype=bool)

    def reset(self, seed=None):
        self.vm.reset(-1) # Reset all instances
        return self._get_obs_all(), {}

    def step(self, actions_p1, actions_p2=None):
        """
        Takes P1 actions and optionally P2 actions.
        """
        a1 = np.asarray(actions_p1, dtype=np.int32)
            
        a2 = None
        if actions_p2 is not None:
            a2 = np.asarray(actions_p2, dtype=np.int32)

        # C++ ネイティブでの一括実行（報酬計算とリセットも内包）
        res = self.vm.step(a1, a2, self._obs_buffer)
        
        # Handle tuple output explicitly
        obs, rewards, terminated, chains = res
        
        # Pack info
        info = {"chains": chains}
        
        return obs, rewards, terminated, self._truncated_buffer, info

    def _get_obs_all(self):
        return self.vm.getObservationsAll(self._obs_buffer)
