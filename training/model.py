"""
Actor-Critic network definitions for PuyotanAI.

Architecture:
  - Input : observation [batch, 2, 5, 6, 14] uint8, cast to float32 internally
  - Output: policy logits + state value (Actor-Critic shared backbone)
  - Backbone is selected via ModelArch enum; callers never depend on internals.
"""
import torch
import torch.nn as nn
from enum import Enum

# ---------------------------------------------------------------------------
# Constants - observation shape
# ---------------------------------------------------------------------------
NUM_ACTIONS  = 22   # 0-5: Up, 6-10: Right, 11-16: Down, 17-21: Left

OBS_PLAYERS  = 2    # self + opponent
OBS_COLORS   = 5    # channels: ojama(0) + color1-4
OBS_COLS     = 6    # board width
OBS_ROWS     = 14   # board height (12 visible + 2 meta rows)

# CNN treats the observation as a multi-channel image: (players * colors) channels
CNN_IN_CHANNELS = OBS_PLAYERS * OBS_COLORS  # 10


# ---------------------------------------------------------------------------
# ModelArch - selects which backbone is instantiated
# ---------------------------------------------------------------------------
class ModelArch(Enum):
    """
    Backbone architecture selector.
    Pass this to PuyotanPolicy; callers (Trainer, Orchestrator) remain unaware
    of internal implementation details.
    """
    MLP = "mlp"
    CNN = "cnn"


# ---------------------------------------------------------------------------
# Backbones (private - not part of the public API)
# ---------------------------------------------------------------------------
class _MLPBackbone(nn.Module):
    """
    Flatten -> fully connected backbone.
    Highest SPS on CPU due to simple operations.
    Spatial relationships between cells are NOT explicitly encoded.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        in_features = OBS_PLAYERS * OBS_COLORS * OBS_COLS * OBS_ROWS  # 840
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, OBS_PLAYERS, OBS_COLORS, OBS_COLS, OBS_ROWS]
        return self.net(x)


class _CNNBackbone(nn.Module):
    """
    Convolutional backbone.
    Treats the observation as (CNN_IN_CHANNELS, OBS_ROWS, OBS_COLS) image so
    that the same spatial pattern (e.g., a chain shape) is recognized regardless
    of its position on the board.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            # Layer 1: detect local adjacency patterns (3x3 receptive field)
            nn.Conv2d(CNN_IN_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Layer 2: detect larger chain shapes
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Feature map size after conv: 64 * OBS_ROWS * OBS_COLS
        conv_out_dim = 64 * OBS_ROWS * OBS_COLS  # 64 * 14 * 6 = 5376
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape [batch, 2, 5, 6, 14] -> [batch, 10, 14, 6] for Conv2d
        b = x.shape[0]
        x = x.reshape(b, CNN_IN_CHANNELS, OBS_ROWS, OBS_COLS)
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# PuyotanPolicy - public interface (the only class callers should reference)
# ---------------------------------------------------------------------------
class PuyotanPolicy(nn.Module):
    """
    Actor-Critic policy network.

    Acts as a facade: Trainer and Orchestrator call only get_action_and_value()
    and are fully decoupled from the chosen backbone implementation.

    Args:
        arch:       ModelArch.MLP (default) or ModelArch.CNN
        hidden_dim: feature dimension for the backbone output and heads
    """
    def __init__(self, arch: ModelArch = ModelArch.MLP, hidden_dim: int = 256):
        super().__init__()
        self.arch = arch

        if arch == ModelArch.MLP:
            self.backbone = _MLPBackbone(hidden_dim)
        elif arch == ModelArch.CNN:
            self.backbone = _CNNBackbone(hidden_dim)
        else:
            raise ValueError(f"Unknown ModelArch: {arch}")

        # Actor and Critic heads are identical regardless of backbone
        self.actor  = nn.Linear(self.backbone.out_dim, NUM_ACTIONS)
        self.critic = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: [batch, 2, 5, 6, 14] float32 tensor
        Returns:
            logits: [batch, NUM_ACTIONS]
            value:  [batch]
        """
        features = self.backbone(obs.float())
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """
        Used by PPOTrainer during rollout and PPO update.
        Returns sampled (or provided) action, log-prob, entropy, and value.
        """
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
