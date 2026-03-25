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
MINIBATCH_SIZE  = 1024
LEARNING_RATE   = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOTrainer:
    def __init__(self, env: PuyotanVectorEnv, num_rollout_steps=128, hidden_dim=128):
        self.env = env
        self.num_envs = env.num_envs
        self.num_steps = num_rollout_steps
        self.model = PuyotanPolicy(hidden_dim=hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-5)

        # Pre-allocate rollout buffers on GPU
        # obs_buf は float32 で保持し、毎ステップの .float() キャストを排除
        self.obs_buf      = torch.zeros((self.num_steps, self.num_envs, 2, 5, 6, 13), dtype=torch.float32, device=DEVICE)
        self.actions_buf  = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long,    device=DEVICE)
        self.logprobs_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        self.values_buf   = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        self.advantages_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        self.returns_buf    = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)

        # rewards / dones は CPU numpy で収集してロールアウト終了後に一括転送
        self.rewards_np = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.dones_np   = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)

        # Latest observation (CPU numpy)
        self.curr_obs, _ = self.env.reset()

    def train(self, p2_policy=None):
        """1イテレーション分の学習を実行。"""
        # 1. ロールアウト収集
        max_chain = self.collect_rollouts(p2_policy)

        # 2. PPOアップデート
        # バッファはすでに正しい型・デバイスにあるため .view() だけで使用可能
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
        max_chain = 0
        obs_cpu = self.curr_obs

        for t in range(self.num_steps):
            # 観測を GPU バッファに保存 (uint8 → float32 キャスト不要)
            self.obs_buf[t].copy_(torch.from_numpy(obs_cpu), non_blocking=True)

            # 方策からのアクション決定
            with torch.inference_mode():
                action, log_prob, _, value = self.model.get_action_and_value(self.obs_buf[t])

            # P2のアクション（推論）
            actions_p2 = None
            if p2_policy is not None:
                actions_p2 = p2_policy.infer(obs_cpu, self.num_envs)

            # 環境ステップ実行
            next_obs, rewards, dones, _, info = self.env.step(action.cpu().numpy(), actions_p2)

            # GPU バッファに書き込み
            self.actions_buf[t]  = action
            self.logprobs_buf[t] = log_prob
            self.values_buf[t]   = value.squeeze()

            # rewards / dones は CPU numpy に直接書き込み（一括転送のため）
            self.rewards_np[t] = rewards
            self.dones_np[t]   = dones

            # 最大連鎖数の追跡
            chains = info.get("chains")
            if chains is not None:
                max_chain = max(max_chain, int(np.max(chains)))

            obs_cpu = next_obs

        self.curr_obs = obs_cpu

        # 最後の状態の価値推定（GAE用）
        with torch.inference_mode():
            last_obs_t = torch.from_numpy(obs_cpu).to(DEVICE, non_blocking=True)
            _, _, _, next_value = self.model.get_action_and_value(last_obs_t)
            next_value = next_value.squeeze().cpu().numpy()

        # GAE 計算 (PyTorchのカーネルディスパッチ遅延を避けるため、CPUのNumPyで一括計算)
        values_cpu = self.values_buf.cpu().numpy()
        advantages_cpu = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.num_steps)):
            not_done = 1.0 - self.dones_np[t]
            next_val = next_value if t == self.num_steps - 1 else values_cpu[t + 1]
            delta    = self.rewards_np[t] + GAMMA * next_val * not_done - values_cpu[t]
            last_gae = delta + GAMMA * LAMBDA * not_done * last_gae
            advantages_cpu[t] = last_gae

        # 計算結果をGPUへ一括書き戻し
        self.advantages_buf.copy_(torch.from_numpy(advantages_cpu), non_blocking=True)
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
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
