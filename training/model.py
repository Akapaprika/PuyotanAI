"""
Actor-Critic ネットワーク定義モジュール（単一責務）

設計方針:
  - 入力: 観測 [batch, 2, 5, 6, 13] の uint8 → float32 に変換してから使用
  - 出力: Policy (行動確率) + Value (状態価値)
  - アーキテクチャ: 畳み込み + 全結合 (Actor-Critic 共有バックボーン)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 行動数: 22 種（0-5: Up, 6-10: Right, 11-16: Down, 17-21: Left）
NUM_ACTIONS = 22

# 入力次元: 2プレイヤー × 5色 × 6列 × 13行 = 780
INPUT_DIM = 2 * 5 * 6 * 13


class PuyotanPolicy(nn.Module):
    """
    Puyo Puyo 用 Actor-Critic ネットワーク。
    観測（盤面状態）から行動確率と状態価値を出力する。
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # 共有バックボーン
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor ヘッド（行動確率）
        self.actor = nn.Linear(hidden_dim, NUM_ACTIONS)

        # Critic ヘッド（状態価値）
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: [batch, 2, 5, 6, 13] の uint8 テンソル
        Returns:
            (行動logits, 状態価値)
        """
        # uint8 → float32 + 正規化
        x = obs.float()
        x = x.view(x.shape[0], -1)  # Flatten: [batch, 780]

        features = self.backbone(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """PPO 学習ループで使用する。行動サンプリングとログ確率を返す。"""
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
