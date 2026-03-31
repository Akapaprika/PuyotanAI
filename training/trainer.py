"""
PPOTrainer: rollout collection and PPO parameter update.

Responsibilities:
  - Allocate rollout buffers (observations, actions, log-probs, values, GAE)
  - collect_rollouts(): run the environment for num_rollout_steps steps
  - ppo_update(): perform PPO gradient updates on collected data
  - save() / load(): checkpoint helpers

Dependencies on PuyotanPolicy are limited to get_action_and_value().
The trainer is agnostic to the backbone architecture (MLP vs CNN).
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from .env import PuyotanVectorEnv
from training.model import PuyotanPolicy, ModelArch
from training.config import (
    GAMMA, LAMBDA, CLIP_EPS, ENTROPY_COEF, VALUE_LOSS_COEF,
    MAX_GRAD_NORM, NUM_EPOCHS, MINIBATCH_SIZE, LEARNING_RATE,
    DEFAULT_HIDDEN_DIM,
)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"


def _to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy. On CPU, .cpu() is a no-op; detach() suffices."""
    return tensor.cpu().numpy() if USE_CUDA else tensor.detach().numpy()


class PPOTrainer:
    def __init__(
        self,
        env: PuyotanVectorEnv,
        num_rollout_steps: int = 128,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        arch: ModelArch = ModelArch.MLP,
    ):
        self.env      = env
        self.num_envs = env.num_envs
        self.num_steps = num_rollout_steps

        self.model = PuyotanPolicy(arch=arch, hidden_dim=hidden_dim).to(DEVICE)
        print(f"[PPOTrainer] backbone={arch.value.upper()}  device={DEVICE}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-5)

        # JIT-compile the model where supported (silently skipped otherwise)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("[PPOTrainer] torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"[PPOTrainer] torch.compile skipped: {e}")

        # ---------------------------------------------------------------------------
        # Rollout buffers - pre-allocated to avoid per-step allocations
        # ---------------------------------------------------------------------------
        S, N = num_rollout_steps, self.num_envs
        self.obs_buf        = torch.zeros((S, N, 2, 5, 6, 14), dtype=torch.float32, device=DEVICE)
        self.actions_buf    = torch.zeros((S, N), dtype=torch.long,    device=DEVICE)
        self.logprobs_buf   = torch.zeros((S, N), dtype=torch.float32, device=DEVICE)
        self.values_buf     = torch.zeros((S, N), dtype=torch.float32, device=DEVICE)
        self.advantages_buf = torch.zeros((S, N), dtype=torch.float32, device=DEVICE)
        self.returns_buf    = torch.zeros((S, N), dtype=torch.float32, device=DEVICE)

        # CPU-side numpy buffers for bulk transfer to GPU after rollout
        self.rewards_np  = np.zeros((S, N), dtype=np.float32)
        self.dones_np    = np.zeros((S, N), dtype=np.float32)
        self.actions_np  = np.zeros((S, N), dtype=np.int32)
        self.logprobs_np = np.zeros((S, N), dtype=np.float32)
        self.values_np   = np.zeros((S, N), dtype=np.float32)

        # Chain statistics per rollout step
        self.chains_max_buf = np.zeros(S, dtype=np.int32)
        self.chain_sums     = np.zeros(S, dtype=np.int32)
        self.chain_counts   = np.zeros(S, dtype=np.int32)
        self.score_sums     = np.zeros(S, dtype=np.int32)
        self.max_per_env    = np.zeros(N, dtype=np.int32)

        # Episodic return tracking
        self.episode_scores           = np.zeros(N, dtype=np.float32)
        self.completed_episode_scores = []

        # Pinned CPU tensor for async GPU upload (effective only with CUDA)
        self.obs_pin = torch.zeros((N, 2, 5, 6, 14), dtype=torch.float32,
                                   pin_memory=USE_CUDA)

        self.curr_obs, _ = self.env.reset()

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------
    def train(self, p2_policy=None) -> dict:
        """Run one rollout + PPO update. Returns a metrics dict."""
        max_chain, avg_max_chain, avg_rew, avg_score = self.collect_rollouts(p2_policy)

        b_obs        = self.obs_buf.view(-1, 2, 5, 6, 14)
        b_actions    = self.actions_buf.view(-1)
        b_log_probs  = self.logprobs_buf.view(-1)
        b_advantages = self.advantages_buf.view(-1)
        b_returns    = self.returns_buf.view(-1)
        b_values     = self.values_buf.view(-1)

        total_loss = sum(
            self.ppo_update(b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values)
            for _ in range(NUM_EPOCHS)
        )

        return {
            "loss":          total_loss / NUM_EPOCHS,
            "max_chain":     max_chain,
            "avg_max_chain": avg_max_chain,
            "avg_reward":    avg_rew,
            "avg_score":     avg_score,
        }

    def save(self, path: str) -> None:
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        torch.save(model.state_dict(), path)

    def load(self, path: str) -> None:
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        model.load_state_dict(torch.load(path, map_location=DEVICE))

    # ---------------------------------------------------------------------------
    # Rollout collection
    # ---------------------------------------------------------------------------
    def collect_rollouts(self, p2_policy=None):
        obs_cpu = self.curr_obs

        # Cache instance attributes locally to avoid repeated attribute lookup
        obs_buf        = self.obs_buf
        obs_pin        = self.obs_pin
        rewards_np     = self.rewards_np
        dones_np       = self.dones_np
        actions_np     = self.actions_np
        logprobs_np    = self.logprobs_np
        values_np      = self.values_np
        chains_max_buf = self.chains_max_buf
        env            = self.env
        model          = self.model
        num_envs       = self.num_envs
        num_steps      = self.num_steps
        from_numpy     = torch.from_numpy
        to_np          = _to_np

        self.max_per_env.fill(0)

        for t in range(num_steps):
            # Async GPU upload via pinned memory
            obs_pin.copy_(from_numpy(obs_cpu))
            obs_buf[t].copy_(obs_pin, non_blocking=True)

            # Sample actions from the current policy
            with torch.inference_mode():
                action, log_prob, _, value = model.get_action_and_value(obs_buf[t])

            act_cpu        = to_np(action)
            actions_np[t]  = act_cpu
            logprobs_np[t] = to_np(log_prob)
            values_np[t]   = to_np(value)

            # Opponent actions: provided policy or mirror P1 in solo mode
            actions_p2 = p2_policy.infer(obs_cpu, num_envs) if p2_policy is not None else act_cpu

            next_obs, rewards, dones, _, info = env.step(act_cpu, actions_p2)

            rewards_np[t] = rewards
            dones_np[t]   = dones

            # Accumulate chain statistics
            chains = info.get("chains")
            if chains is not None:
                chains_max_buf[t] = int(np.max(chains))
                self.max_per_env  = np.maximum(self.max_per_env, chains)
                nonzero = chains[chains > 0]
                self.chain_sums[t]   = int(np.sum(nonzero))
                self.chain_counts[t] = len(nonzero)
            else:
                chains_max_buf[t]    = 0
                self.chain_sums[t]   = 0
                self.chain_counts[t] = 0

            obs_cpu = next_obs

            # Accumulate step-level score deltas
            scores = info.get("scores")
            if scores is not None:
                self.episode_scores += scores

            for i in range(num_envs):
                if dones[i]:
                    self.completed_episode_scores.append(self.episode_scores[i])
                    self.episode_scores[i] = 0.0

        self.curr_obs = obs_cpu

        max_chain     = int(np.max(chains_max_buf))
        avg_max_chain = float(np.mean(self.max_per_env))
        avg_reward    = float(np.mean(rewards_np))

        if self.completed_episode_scores:
            avg_score = float(np.mean(self.completed_episode_scores))
            self.completed_episode_scores = []
        else:
            avg_score = float(np.mean(self.episode_scores))

        # Bulk-transfer rollout tensors to GPU
        self.actions_buf.copy_(from_numpy(actions_np),  non_blocking=True)
        self.logprobs_buf.copy_(from_numpy(logprobs_np), non_blocking=True)
        self.values_buf.copy_(from_numpy(values_np),    non_blocking=True)

        # Bootstrap value estimate for the last observed state (used by GAE)
        with torch.inference_mode():
            obs_pin.copy_(from_numpy(obs_cpu))
            last_obs_t = obs_pin.to(DEVICE, non_blocking=True)
            _, _, _, next_value = model.get_action_and_value(last_obs_t)
            next_value = to_np(next_value)

        # GAE computed in C++ to eliminate Python loops
        from training.env import p
        advantages_cpu = p.compute_gae(
            rewards_np, values_np, dones_np, next_value,
            float(GAMMA), float(LAMBDA)
        )

        self.advantages_buf.copy_(from_numpy(advantages_cpu), non_blocking=True)
        self.returns_buf.copy_(self.advantages_buf + self.values_buf)

        return max_chain, avg_max_chain, avg_reward, avg_score

    # ---------------------------------------------------------------------------
    # PPO update
    # ---------------------------------------------------------------------------
    def ppo_update(self, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values) -> float:
        n        = b_obs.shape[0]
        indices  = torch.randperm(n, device=DEVICE)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_loss = 0.0
        for start in range(0, n, MINIBATCH_SIZE):
            idx = indices[start:start + MINIBATCH_SIZE]
            _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                b_obs[idx], b_actions[idx]
            )
            adv     = b_advantages[idx]
            ratio   = torch.exp(new_log_prob - b_log_probs[idx])
            loss_pi = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            ).mean()
            loss_v   = 0.5 * ((new_value - b_returns[idx]) ** 2).mean()
            loss_ent = -entropy.mean()

            loss = loss_pi + VALUE_LOSS_COEF * loss_v + ENTROPY_COEF * loss_ent
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss
