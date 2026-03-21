"""
PPO 学習ループ（Phase 1: Stage 1 - PASS 相手）

方針:
  - 10,000 並列環境（PuyotanVectorEnv）で高速に経験を収集
  - P2 は PASS（何もしない）固定。まず連鎖構築を学ばせる
  - 報酬は training/reward.py に委任
  - 一定ステップごとにチェックポイントを保存

使用方法:
    python -m training.trainer
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ───────────────── パス設定 ─────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR / "native" / "dist", BASE_DIR / "native" / "build_Release" / "Release"]:
    if d.exists():
        sys.path.insert(0, str(d))
        break

import puyotan_native as p
from puyotan_env import PuyotanVectorEnv, ActionMapper
from training.model import PuyotanPolicy

# ───────────────── ハイパーパラメータ ─────────────────
NUM_ENVS        = 1024     # 並列環境数（メモリに応じて調整）
NUM_STEPS       = 64       # 1回の rollout ステップ数
TOTAL_STEPS     = 5_000_000
LEARNING_RATE   = 3e-4
GAMMA           = 0.99     # 割引率
LAMBDA          = 0.95     # GAE λ
CLIP_EPS        = 0.2      # PPO クリップ係数
ENTROPY_COEF    = 0.01     # エントロピーボーナス係数
VALUE_LOSS_COEF = 0.5      # 価値損失係数
MAX_GRAD_NORM   = 0.5      # 勾配クリッピング
NUM_EPOCHS      = 4        # 1 rollout あたりの学習エポック数
MINIBATCH_SIZE  = 256      # ミニバッチサイズ
SAVE_INTERVAL   = 100_000  # チェックポイント保存間隔（ステップ数）
MODEL_DIR       = BASE_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=LAMBDA):
    """Generalized Advantage Estimation (GAE)。"""
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * (next_value if t == len(rewards) - 1 else values[t + 1]) * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def rollout(env: PuyotanVectorEnv, policy: PuyotanPolicy, obs_buf):
    """
    NUM_STEPS ステップ分の経験を収集して返す。
    P2 は常に PASS。
    """
    all_obs      = []
    all_actions  = []
    all_log_probs = []
    all_rewards  = []
    all_dones    = []
    all_values   = []

    obs = obs_buf  # [NUM_ENVS, 2, 5, 6, 13] uint8

    for _ in range(NUM_STEPS):
        obs_t = torch.from_numpy(obs).to(DEVICE)
        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(obs_t)

        # P1 の行動をセット
        match_indices = list(range(env.num_envs))
        p1_ids = [0] * env.num_envs
        p1_actions = []
        for a in action.cpu().numpy():
            col, rot = ActionMapper.get(int(a))
            p1_actions.append(p.Action(p.ActionType.PUT, col, rot))
        env.vm.set_actions(match_indices, p1_ids, p1_actions)

        # P2 は PASS（Stage 1）
        p2_ids = [1] * env.num_envs
        p2_actions = [p.Action(p.ActionType.PASS, 0, p.Rotation.Up)] * env.num_envs
        env.vm.set_actions(match_indices, p2_ids, p2_actions)

        # シミュレーション進行
        env.vm.step_until_decision()

        # 観測・報酬・終了フラグ収集
        next_obs = env.vm.get_observations_all()  # uint8

        rewards = np.zeros(env.num_envs, dtype=np.float32)
        dones   = np.zeros(env.num_envs, dtype=np.float32)

        for i in range(env.num_envs):
            m = env.vm.get_match(i)
            p1 = m.getPlayer(0)
            rewards[i] = p1.score * 0.002  # スコア増加を報酬に

            if m.status != p.MatchStatus.PLAYING:
                if m.status == p.MatchStatus.WIN_P1:
                    rewards[i] += 10.0
                elif m.status == p.MatchStatus.WIN_P2:
                    rewards[i] -= 10.0
                dones[i] = 1.0
                # 終了した環境はリセット
                env.vm.reset(i)
                env.vm.get_match(i).start()
                env.vm.get_match(i).step_until_decision()

        all_obs.append(obs_t)
        all_actions.append(action)
        all_log_probs.append(log_prob)
        all_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=DEVICE))
        all_dones.append(torch.tensor(dones, dtype=torch.float32, device=DEVICE))
        all_values.append(value)

        obs = next_obs

    # 最後の状態の価値推定
    with torch.no_grad():
        _, _, _, next_value = policy.get_action_and_value(
            torch.from_numpy(obs).to(DEVICE)
        )

    all_values_t = torch.stack(all_values)           # [T, N]
    all_rewards_t = torch.stack(all_rewards)          # [T, N]
    all_dones_t   = torch.stack(all_dones)            # [T, N]

    advantages, returns = compute_gae(all_rewards_t, all_values_t, all_dones_t, next_value)

    # フラット化して返す
    b_obs      = torch.cat(all_obs).view(-1, *obs_t.shape[1:])  # [T*N, ...]
    b_actions  = torch.cat(all_actions).view(-1)
    b_log_probs = torch.cat(all_log_probs).view(-1)
    b_advantages = advantages.view(-1)
    b_returns  = returns.view(-1)
    b_values   = all_values_t.view(-1)

    return b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values, obs


def ppo_update(policy, optimizer, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values):
    """PPO ミニバッチ更新。"""
    n = b_obs.shape[0]
    indices = torch.randperm(n, device=DEVICE)
    total_loss = 0.0

    for start in range(0, n, MINIBATCH_SIZE):
        idx = indices[start:start + MINIBATCH_SIZE]
        obs_mb    = b_obs[idx]
        act_mb    = b_actions[idx]
        old_lp_mb = b_log_probs[idx]
        adv_mb    = b_advantages[idx]
        ret_mb    = b_returns[idx]
        val_mb    = b_values[idx]

        _, new_log_prob, entropy, new_value = policy.get_action_and_value(obs_mb, act_mb)

        # アドバンテージの正規化
        adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

        # PPO クリップ損失（Policy Loss）
        ratio = torch.exp(new_log_prob - old_lp_mb)
        loss_pi = -torch.min(
            ratio * adv_mb,
            torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
        ).mean()

        # 価値損失（Value Loss）
        loss_v = 0.5 * ((new_value - ret_mb) ** 2).mean()

        # エントロピーボーナス
        loss_entropy = -entropy.mean()

        loss = loss_pi + VALUE_LOSS_COEF * loss_v + ENTROPY_COEF * loss_entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def train():
    print(f"Device: {DEVICE}")
    print(f"Num envs: {NUM_ENVS}, Steps per rollout: {NUM_STEPS}")
    print(f"Total steps: {TOTAL_STEPS:,}")
    MODEL_DIR.mkdir(exist_ok=True)

    # 環境・モデル・オプティマイザ初期化
    env = PuyotanVectorEnv(NUM_ENVS, base_seed=1)
    obs, _ = env.reset()

    policy = PuyotanPolicy(hidden_dim=256).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=1e-5)

    global_step = 0
    update_count = 0
    start_time = time.perf_counter()

    print("Starting training (Stage 1: PASS opponent)...")
    while global_step < TOTAL_STEPS:
        # Rollout（経験収集）
        b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values, obs = \
            rollout(env, policy, obs)

        global_step += NUM_ENVS * NUM_STEPS
        update_count += 1

        # PPO 更新
        for _ in range(NUM_EPOCHS):
            loss = ppo_update(
                policy, optimizer,
                b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values
            )

        if update_count % 10 == 0:
            elapsed = time.perf_counter() - start_time
            fps = global_step / elapsed
            mean_ret = b_returns.mean().item()
            print(f"Step: {global_step:>10,} | FPS: {fps:>8.0f} | "
                  f"MeanReturn: {mean_ret:>7.3f} | Loss: {loss:.4f}")

        # チェックポイント保存
        if global_step % SAVE_INTERVAL < NUM_ENVS * NUM_STEPS:
            ckpt_path = MODEL_DIR / f"puyotan_step{global_step}.pt"
            torch.save(policy.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # 最終モデル保存
    final_path = MODEL_DIR / "puyotan_final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    train()
