import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from .env import PuyotanVectorEnv
from training.model import PuyotanPolicy

# ───────────────── ハイパーパラメータ ─────────────────
GAMMA           = 0.99
LAMBDA          = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM   = 0.5
NUM_EPOCHS      = 2
MINIBATCH_SIZE  = 2048
LEARNING_RATE   = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

# CPU デバイスでは .cpu() は no-op なので .numpy() だけで済む
def _to_np(tensor):
    return tensor.cpu().numpy() if USE_CUDA else tensor.detach().numpy()

class PPOTrainer:
    def __init__(self, env: PuyotanVectorEnv, num_rollout_steps=128, hidden_dim=128):
        self.env = env
        self.num_envs = env.num_envs
        self.num_steps = num_rollout_steps
        self.model = PuyotanPolicy(hidden_dim=hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-5)

        # torch.compile で推論・学習カーネルを JIT 最適化（対応環境のみ）
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("[torch.compile] Model compiled with reduce-overhead mode.")
        except Exception as e:
            print(f"[torch.compile] Skipped (not supported on this platform): {e}")

        # Pre-allocate rollout buffers on GPU
        self.obs_buf        = torch.zeros((self.num_steps, self.num_envs, 2, 5, 6, 13), dtype=torch.float32, device=DEVICE)
        self.actions_buf    = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long,    device=DEVICE)
        self.logprobs_buf   = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        self.values_buf     = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        self.advantages_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        self.returns_buf    = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)

        # CPU numpy バッファ（一括転送用）
        self.rewards_np  = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.dones_np    = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.actions_np  = np.zeros((self.num_steps, self.num_envs), dtype=np.int32)
        self.logprobs_np = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.values_np   = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        # chains はロールアウト最後に一括 max を取る
        self.chains_max_buf = np.zeros(self.num_steps, dtype=np.int32)

        # ピン留め CPU テンソル（CUDA 環境のみ有効）
        self._pin = USE_CUDA
        self.obs_pin = torch.zeros((self.num_envs, 2, 5, 6, 13), dtype=torch.float32,
                                   pin_memory=self._pin)

        # Latest observation (CPU numpy)
        self.curr_obs, _ = self.env.reset()

    def train(self, p2_policy=None):
        """1イテレーション分の学習を実行。"""
        max_chain = self.collect_rollouts(p2_policy)

        b_obs        = self.obs_buf.view(-1, 2, 5, 6, 13)
        b_actions    = self.actions_buf.view(-1)
        b_log_probs  = self.logprobs_buf.view(-1)
        b_advantages = self.advantages_buf.view(-1)
        b_returns    = self.returns_buf.view(-1)
        b_values     = self.values_buf.view(-1)

        total_loss = 0
        for _ in range(NUM_EPOCHS):
            loss = self.ppo_update(b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values)
            total_loss += loss

        return {"loss": total_loss / NUM_EPOCHS, "max_chain": max_chain}

    def collect_rollouts(self, p2_policy=None):
        obs_cpu = self.curr_obs

        # ローカル変数キャッシュ（Pythonアトリビュート解決を排除）
        obs_buf = self.obs_buf
        obs_pin = self.obs_pin
        rewards_np = self.rewards_np
        dones_np = self.dones_np
        actions_np = self.actions_np
        logprobs_np = self.logprobs_np
        values_np = self.values_np
        chains_max_buf = self.chains_max_buf
        env = self.env
        model = self.model
        num_envs = self.num_envs
        num_steps = self.num_steps
        from_numpy = torch.from_numpy
        to_np = _to_np

        for t in range(num_steps):
            # ピン留めメモリ経由で非同期 GPU 転送
            obs_pin.copy_(from_numpy(obs_cpu))
            obs_buf[t].copy_(obs_pin, non_blocking=True)

            # 方策からのアクション決定
            with torch.inference_mode():
                action, log_prob, _, value = model.get_action_and_value(obs_buf[t])

            # CPU numpy に集積（.cpu() 呼び出しを CPU デバイスではスキップ）
            act_cpu = to_np(action)
            actions_np[t]  = act_cpu
            logprobs_np[t] = to_np(log_prob)
            values_np[t]   = to_np(value)

            # P2のアクション（推論）
            actions_p2 = None
            if p2_policy is not None:
                actions_p2 = p2_policy.infer(obs_cpu, num_envs)

            # 環境ステップ実行（actions_p1 は既に int64 numpy。int32 変換は env 側で不要化済み）
            next_obs, rewards, dones, _, info = env.step(act_cpu, actions_p2)

            rewards_np[t] = rewards
            dones_np[t]   = dones

            # chains の max は毎ステップ計算せず、各ステップの max だけ記録
            chains = info.get("chains")
            chains_max_buf[t] = int(np.max(chains)) if chains is not None else 0

            obs_cpu = next_obs

        self.curr_obs = obs_cpu

        # chains の全体 max はロールアウト終了後に1回だけ計算
        max_chain = int(np.max(chains_max_buf))

        # ロールアウトデータを一括で GPU に転送
        self.actions_buf.copy_(from_numpy(actions_np), non_blocking=True)
        self.logprobs_buf.copy_(from_numpy(logprobs_np), non_blocking=True)
        self.values_buf.copy_(from_numpy(values_np), non_blocking=True)

        # 最後の状態の価値推定（GAE用）
        with torch.inference_mode():
            obs_pin.copy_(from_numpy(obs_cpu))
            last_obs_t = obs_pin.to(DEVICE, non_blocking=True)
            _, _, _, next_value = model.get_action_and_value(last_obs_t)
            next_value = to_np(next_value)

        # GAE 計算（CPU NumPy で一括計算）
        advantages_cpu = np.zeros((num_steps, num_envs), dtype=np.float32)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(num_steps)):
            not_done = 1.0 - dones_np[t]
            next_val = next_value if t == num_steps - 1 else values_np[t + 1]
            delta    = rewards_np[t] + GAMMA * next_val * not_done - values_np[t]
            last_gae = delta + GAMMA * LAMBDA * not_done * last_gae
            advantages_cpu[t] = last_gae

        self.advantages_buf.copy_(from_numpy(advantages_cpu), non_blocking=True)
        self.returns_buf.copy_(self.advantages_buf + self.values_buf)
        return max_chain

    def ppo_update(self, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values):
        n = b_obs.shape[0]
        indices = torch.randperm(n, device=DEVICE)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_loss = 0
        for start in range(0, n, MINIBATCH_SIZE):
            idx = indices[start:start + MINIBATCH_SIZE]
            _, new_log_prob, entropy, new_value = self.model.get_action_and_value(b_obs[idx], b_actions[idx])

            adv = b_advantages[idx]

            ratio    = torch.exp(new_log_prob - b_log_probs[idx])
            loss_pi  = -torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv).mean()
            loss_v   = 0.5 * ((new_value - b_returns[idx]) ** 2).mean()
            loss_ent = -entropy.mean()

            loss = loss_pi + VALUE_LOSS_COEF * loss_v + ENTROPY_COEF * loss_ent
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def save(self, path):
        model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        torch.save(model.state_dict(), path)

    def load(self, path):
        model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model.load_state_dict(torch.load(path, map_location=DEVICE))
