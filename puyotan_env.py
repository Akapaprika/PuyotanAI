import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path
from training.reward import calculate_reward

# Add the native library path
BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "native" / "dist"
RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"

for d in [DIST_DIR, RELEASE_DIR]:
    if d.exists():
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

class PuyotanEnv(gym.Env):
    """
    Puyo Puyo environment for a single match (Self vs Random/Fixed P2).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None):
        super().__init__()
        self.action_space = spaces.Discrete(len(ActionMapper.ACTIONS))
        # 2 players, 5 colors, 6 width, 13 height
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, 5, 6, 13), dtype=np.uint8
        )
        self.match = None
        self._seed = seed if seed is not None else 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            
        self.match = p.PuyotanMatch(self._seed)
        self.match.start()
        
        # Advance to first decision
        self.match.step_until_decision()
        
        return self._get_obs(), {}

    def step(self, action):
        col, rot = ActionMapper.get(action)
        
        # 1. Set self (P1) action
        self.match.setAction(0, p.Action(p.ActionType.PUT, col, rot))
        
        # 2. Set opponent (P2) action (Simple dummy: PASS or Random)
        # For a standard gym env, P2 is usually part of the env.
        # Here we make P2 just PASS to allow P1 to play alone for now.
        status_mask = self.match.step_until_decision() # This won't advance if P2 is unset
        if status_mask & 2: # P2 needs action
            self.match.setAction(1, p.Action(p.ActionType.PASS, 0, p.Rotation.Up))

        # 3. Step simulation
        self.match.stepNextFrame()
        
        # 4. Advance until next decision or game over
        self.match.step_until_decision()
        
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = (self.match.status != p.MatchStatus.PLAYING)
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # We need to collect observations for both players
        p1 = self.match.getPlayer(0)
        p2 = self.match.getPlayer(1)
        
        obs = np.zeros((2, 5, 6, 13), dtype=np.uint8)
        # to_obs_flat returns (5, 6, 13) uint8
        obs[0] = p1.to_obs_flat()
        obs[1] = p2.to_obs_flat()
        return obs

    def _calculate_reward(self):
        # This is a simplified reward based on P1's score growth
        # In a real scenario, you might want to track score delta
        p1 = self.match.getPlayer(0)
        # Reward for sending ojama
        reward = p1.score / 70.0 
        
        if self.match.status == p.MatchStatus.WIN_P1:
            reward += 10.0
        elif self.match.status == p.MatchStatus.WIN_P2:
            reward -= 10.0
            
        return float(reward)

    def render(self):
        pass

class PuyotanVectorEnv:
    """
    Hig-performance Vectorized environment for Puyotan!
    Directly wraps PuyotanVectorMatch for massive parallel training.
    """
    def __init__(self, num_envs=1, base_seed=1):
        print(f"DEBUG: PuyotanVectorEnv __init__ with num_envs={num_envs}, base_seed={base_seed}")
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.vm = p.PuyotanVectorMatch(num_envs, base_seed)
        print("DEBUG: PuyotanVectorMatch created.")
        
        # Ensure ActionMapper is ready
        ActionMapper.initialize()
        self.action_space = spaces.Discrete(len(ActionMapper.ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, 5, 6, 13), dtype=np.uint8
        )
        self.base_seed = base_seed
        
        # 報酬計算用の前回状態を保持
        self.prev_scores = np.zeros(num_envs, dtype=np.int32)
        self.prev_ojama = np.zeros(num_envs, dtype=np.int32)

    def reset(self, seed=None):
        # We don't support individual reset yet, just reset all
        self.vm.reset(-1) # Reset all
        for i in range(self.num_envs):
            match = self.vm.get_match(i)
            match.start()
            # Advance to first decision
            match.step_until_decision()
            
            self.prev_scores[i] = match.getPlayer(0).score
            self.prev_ojama[i] = match.getPlayer(0).active_ojama

        return self._get_obs_all(), {}

    def step(self, actions_p1, actions_p2=None):
        """
        Takes P1 actions and optionally P2 actions.
        If actions_p2 is None, P2 will PASS.
        """
        # 1. Set P1 actions
        match_indices = list(range(self.num_envs))
        p1_ids = [0] * self.num_envs
        p1_acts = []
        for a in actions_p1:
            col, rot = ActionMapper.get(a)
            p1_acts.append(p.Action(p.ActionType.PUT, col, rot))
        self.vm.set_actions(match_indices, p1_ids, p1_acts)

        # 2. Set P2 actions (if provided, otherwise PASS)
        p2_ids = [1] * self.num_envs
        p2_acts = []
        if actions_p2 is not None:
            for a in actions_p2:
                col, rot = ActionMapper.get(a)
                p2_acts.append(p.Action(p.ActionType.PUT, col, rot))
        else:
            p2_acts = [p.Action(p.ActionType.PASS, 0, p.Rotation.Up)] * self.num_envs
        self.vm.set_actions(match_indices, p2_ids, p2_acts)
        
        # 3. Step all environments & Collect rewards/dones
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        
        for i in range(self.num_envs):
            match = self.vm.get_match(i)
            match.step_until_decision()
            
            # 報酬計算 (P1視点)
            rewards[i] = calculate_reward(
                self.prev_scores[i], match.getPlayer(0).score,
                self.prev_ojama[i], match.getPlayer(0).active_ojama,
                match.status
            )
            
            # 状態更新
            self.prev_scores[i] = match.getPlayer(0).score
            self.prev_ojama[i] = match.getPlayer(0).active_ojama
            
            # 終了判定
            if match.status != p.MatchStatus.PLAYING:
                terminated[i] = True
                # 自動リセット（強化学習の標準的な挙動）
                match.reset()
                match.start()
                match.step_until_decision()
                self.prev_scores[i] = match.getPlayer(0).score
                self.prev_ojama[i] = match.getPlayer(0).active_ojama
        
        obs = self._get_obs_all()
        return obs, rewards, terminated, truncated, {}

    def _get_obs_all(self):
        res = self.vm.get_observations_all()
        # Explicit type check for debugging
        if not isinstance(res, np.ndarray):
            try:
                # If it's a tuple/list, the actual array is likely the 0th element
                return res[0]
            except:
                pass
        return res
