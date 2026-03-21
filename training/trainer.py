import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import puyotan_native as p
from puyotan_env import PuyotanVectorEnv, ActionMapper
from training.model import PuyotanPolicy

# ───────────────── ハイパーパラメータ ─────────────────
GAMMA           = 0.99
LAMBDA          = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM   = 0.5
NUM_EPOCHS      = 4
MINIBATCH_SIZE  = 256
LEARNING_RATE   = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOTrainer:
    def __init__(self, env: PuyotanVectorEnv, hidden_dim=256):
        self.env = env
        self.num_envs = env.num_envs
        self.model = PuyotanPolicy(hidden_dim=hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-5)
        try:
            self.obs_buf, _ = self.env.reset()
        except ValueError as e:
            print(f"FAILED AT env.reset(): {e}")
            res = self.env.reset()
            print(f"ACTUAL RESET RETURN TYPE: {type(res)}")
            print(f"ACTUAL RESET RETURN VALUE: {res}")
            raise e

    def train(self, num_steps=128, p2_policy=None):
        """1イテレーション分の学習を実行。"""
        # 1. ロールアウト収集
        b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values = \
            self.collect_rollouts(num_steps, p2_policy)

        # 2. PPOアップデート
        total_loss = 0
        for _ in range(NUM_EPOCHS):
            loss = self.ppo_update(b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values)
            total_loss += loss
            
        return {"loss": total_loss / NUM_EPOCHS}

    def collect_rollouts(self, num_steps, p2_policy=None):
        all_obs, all_actions, all_log_probs, all_rewards, all_dones, all_values = [], [], [], [], [], []

        obs = self.obs_buf
        for _ in range(num_steps):
            if not isinstance(obs, np.ndarray):
                print(f"CRITICAL: obs is NOT ndarray! type={type(obs)}")
            obs_t = torch.from_numpy(obs).to(DEVICE)
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(obs_t)

            # P2のアクションを決定
            actions_p2 = None
            if p2_policy is not None:
                # C++ OnnxPolicy を使用
                actions_p2 = p2_policy.infer(obs, self.num_envs)

            # 環境を1ステップ進める
            try:
                next_obs, rewards, dones, _, _ = self.env.step(action.cpu().numpy(), actions_p2)
            except Exception as e:
                print(f"FAILED AT env.step: {e}")
                obs_dump = self.env._get_obs_all()
                print(f"OBS DUMP Type: {type(obs_dump)}")
                raise e

            all_obs.append(obs_t)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=DEVICE))
            all_dones.append(torch.tensor(dones, dtype=torch.float32, device=DEVICE))
            all_values.append(value)

            obs = next_obs

        self.obs_buf = obs # 次回の開始点として保持

        # 最後の状態の価値推定
        with torch.no_grad():
            _, _, _, next_value = self.model.get_action_and_value(torch.from_numpy(obs).to(DEVICE))

        all_values_t = torch.stack(all_values)
        all_rewards_t = torch.stack(all_rewards)
        all_dones_t = torch.stack(all_dones)

        # GAE計算
        advantages = torch.zeros_like(all_rewards_t)
        last_gae = 0.0
        for t in reversed(range(num_steps)):
            not_done = 1.0 - all_dones_t[t]
            next_val = next_value if t == num_steps - 1 else all_values_t[t+1]
            delta = all_rewards_t[t] + GAMMA * next_val * not_done - all_values_t[t]
            last_gae = delta + GAMMA * LAMBDA * not_done * last_gae
            advantages[t] = last_gae
        
        returns = advantages + all_values_t

        return (
            torch.cat(all_obs).view(-1, *obs_t.shape[1:]),
            torch.cat(all_actions).view(-1),
            torch.cat(all_log_probs).view(-1),
            advantages.view(-1),
            returns.view(-1),
            all_values_t.view(-1)
        )

    def ppo_update(self, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values):
        n = b_obs.shape[0]
        indices = torch.randperm(n, device=DEVICE)
        total_loss = 0
        for start in range(0, n, MINIBATCH_SIZE):
            idx = indices[start:start + MINIBATCH_SIZE]
            _, new_log_prob, entropy, new_value = self.model.get_action_and_value(b_obs[idx], b_actions[idx])
            
            adv = b_advantages[idx]
            if len(adv) > 1 and adv.std() > 0:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            ratio = torch.exp(new_log_prob - b_log_probs[idx])
            loss_pi = -torch.min(ratio * adv, torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv).mean()
            loss_v = 0.5 * ((new_value - b_returns[idx])**2).mean()
            loss_ent = -entropy.mean()
            
            loss = loss_pi + VALUE_LOSS_COEF * loss_v + ENTROPY_COEF * loss_ent
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
