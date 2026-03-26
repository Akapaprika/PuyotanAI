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
except ImportError:
    p = None

# 行動数定数（ActionMapper の結果と一致）
NUM_ACTIONS = 22

class PuyotanVectorEnv:
    """
    High-performance Vectorized environment for Puyotan!
    Directly wraps PuyotanVectorMatch for massive parallel training.
    """
    def __init__(self, num_envs=1, base_seed=1):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.vm = p.PuyotanVectorMatch(num_envs, base_seed)
        self._obs_buffer = np.zeros((num_envs, 2, 5, 6, 13), dtype=np.uint8)
        # 事前確保: step の戻り値で使い回す
        self._truncated = np.zeros(num_envs, dtype=bool)
        self._info = {"chains": None}

    def reset(self, seed=None):
        self.vm.reset(-1)
        return self._get_obs_all(), {}

    def step(self, actions_p1, actions_p2=None):
        """
        Takes P1 actions (np.int32 ndarray) and optionally P2 actions.
        Returns (obs, rewards, terminated, truncated, info).
        """
        # C++ ネイティブでの一括実行（報酬計算とリセットも内包）
        obs, rewards, terminated, chains = self.vm.step(
            actions_p1, actions_p2, self._obs_buffer
        )
        self._info["chains"] = chains
        return obs, rewards, terminated, self._truncated, self._info

    def _get_obs_all(self):
        return self.vm.getObservationsAll(self._obs_buffer)
