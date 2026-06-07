"""
ONNX Export Script for PuyotanAI ResNet/CNN/MLP policies.

Usage:
    python -m orchestrator.export_onnx --arch resnet
    python -m orchestrator.export_onnx --arch resnet --channels 32 --num_blocks 2 --hidden_dim 128
    python -m orchestrator.export_onnx --arch cnn
    python -m orchestrator.export_onnx --arch mlp

Output:
    models/<arch>/puyotan_<arch>.onnx

ONNX I/O spec (must match OnnxPolicy in C++):
    Input  "obs"    : uint8  [N, 2, 5, 6, 14]
    Output "logits" : float32 [N, 22]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants (must match C++ constants.hpp)
# ---------------------------------------------------------------------------
kObsPlayers = 2
kObsColors  = 5
kObsCols    = 6
kObsRows    = 14
kCnnInChannels = kObsPlayers * kObsColors  # 10
kNumActions = 22

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Python mirror of C++ ResNet architecture (must exactly match resnet_policy.cpp)
# ---------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return torch.relu(x)


class ResNetBackbone(nn.Module):
    def __init__(self, hidden_dim: int, channels: int = 32, num_blocks: int = 2):
        super().__init__()
        self.conv_in = nn.Conv2d(kCnnInChannels, channels, 3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(channels)
        self.blocks  = nn.Sequential(*[ResNetBlock(channels) for _ in range(num_blocks)])
        fc_in        = channels * (1 + kObsCols)  # GAP + ColMax concat
        self.fc      = nn.Linear(fc_in, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.reshape(b, kCnnInChannels, kObsCols, kObsRows)  # [B, 10, 6, 14]
        x = x.transpose(2, 3).contiguous()                     # [B, 10, 14, 6]
        x = torch.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        # Stream 1: GAP → [B, channels]
        gap = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        # Stream 2: Column max pool → [B, channels * kObsCols]
        col = x.max(dim=2).values.flatten(1)
        combined = torch.cat([gap, col], dim=1)
        return torch.relu(self.fc(combined))


class ResNetPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128, channels: int = 32, num_blocks: int = 2):
        super().__init__()
        self.backbone = ResNetBackbone(hidden_dim, channels, num_blocks)
        self.actor    = nn.Linear(hidden_dim, kNumActions)
        self.critic   = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        return self.actor(features), self.critic(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Python mirror of C++ CNN architecture (matches cnn_policy.cpp)
# ---------------------------------------------------------------------------
class CNNBackbone(nn.Module):
    def __init__(self, hidden_dim: int, channels: int = 24):
        super().__init__()
        self.conv1 = nn.Conv2d(kCnnInChannels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        fc_in = channels * kObsCols * kObsRows
        self.fc     = nn.Linear(fc_in, hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        # Matches C++ CNNBackboneImpl::forward reshape exactly
        x = obs.reshape(b, kCnnInChannels, kObsRows, kObsCols).float() # [B, 10, 14, 6]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(b, -1)
        x = torch.relu(self.fc(x))
        return x


class CNNPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128, channels: int = 24):
        super().__init__()
        self.backbone = CNNBackbone(hidden_dim, channels)
        self.actor    = nn.Linear(hidden_dim, kNumActions)
        self.critic   = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        return self.actor(features), self.critic(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Python mirror of C++ MLP architecture (matches mlp_policy.cpp)
# ---------------------------------------------------------------------------
class MLPBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        obs_dim = kObsPlayers * kObsColors * kObsCols * kObsRows  # 840
        self.fc1    = nn.Linear(obs_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1).float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class MLPPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.backbone = MLPBackbone(hidden_dim)
        self.actor    = nn.Linear(hidden_dim, kNumActions)
        self.critic   = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        return self.actor(features), self.critic(features).squeeze(-1)


# ---------------------------------------------------------------------------
# ONNX Export wrapper: uint8 input → float cast → logits output
# (matches OnnxPolicy::infer in C++ which passes uint8 directly)
# ---------------------------------------------------------------------------
class PolicyForExport(nn.Module):
    """
    Wraps a policy so that:
      - Input  "obs"    : uint8  [N, 2, 5, 6, 14]
      - Output "logits" : float32 [N, 22]
    """
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()           # uint8 → float32
        logits, _ = self.policy(x)
        return logits             # [N, 22]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def export(
    arch: str,
    hidden_dim: int,
    channels: int,
    num_blocks: int,
    pt_path: Path | None = None,
    onnx_path: Path | None = None,
) -> None:
    arch_dir  = BASE_DIR / "models" / arch
    
    if pt_path is None:
        pt_path = arch_dir / "puyotan_solo_latest.pt"
        if not pt_path.exists():
            fallback_pt = arch_dir / "puyotan_latest.pt"
            if fallback_pt.exists():
                pt_path = fallback_pt

    if onnx_path is None:
        onnx_path = arch_dir / f"puyotan_{arch}.onnx"

    if not pt_path.exists():
        print(f"[ERROR] Checkpoint not found: {pt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {pt_path}")

    # Build model
    if arch == "resnet":
        policy = ResNetPolicy(hidden_dim, channels, num_blocks)
    elif arch == "cnn":
        policy = CNNPolicy(hidden_dim, channels)
    elif arch in ("mlp", "light_mlp"):
        policy = MLPPolicy(hidden_dim)
    else:
        print(f"[ERROR] Unknown arch: {arch}")
        sys.exit(1)

    # LibTorch の OutputArchive は TorchScript 互換の ZIP 形式で保存される。
    # Python 2.6+ では torch.load() がデフォルト weights_only=True になったため、
    # torch.jit.load() で ScriptModule として読み込み、state_dict を抽出する。
    try:
        loaded = torch.jit.load(str(pt_path), map_location="cpu")
        state_dict = loaded.state_dict()
        print("  (loaded as TorchScript module)")
    except Exception:
        loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        state_dict = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
        print("  (loaded as state dict)")

    policy.load_state_dict(state_dict)
    policy.eval()

    export_model = PolicyForExport(policy)
    export_model.eval()

    # Dummy input: uint8 [1, 2, 5, 6, 14]
    dummy = torch.zeros(1, kObsPlayers, kObsColors, kObsCols, kObsRows, dtype=torch.uint8)

    print(f"Exporting to: {onnx_path}")
    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[OK] Exported: {onnx_path}")
    print(f"     Input : obs    uint8  [N, {kObsPlayers}, {kObsColors}, {kObsCols}, {kObsRows}]")
    print(f"     Output: logits float32 [N, {kNumActions}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PuyotanAI policy to ONNX")
    parser.add_argument("--arch",       type=str, default="resnet",
                        choices=["mlp", "cnn", "resnet", "light_mlp"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--channels",   type=int, default=32,
                        help="CNN/ResNet channel count (ignored for MLP)")
    parser.add_argument("--num_blocks", type=int, default=2,
                        help="ResNet block count (ignored for CNN/MLP)")
    parser.add_argument("--pt_path",    type=str, default=None,
                        help="Path to input .pt file (optional)")
    parser.add_argument("--onnx_path",  type=str, default=None,
                        help="Path to output .onnx file (optional)")
    args = parser.parse_args()

    pt_path_obj = Path(args.pt_path) if args.pt_path else None
    onnx_path_obj = Path(args.onnx_path) if args.onnx_path else None

    export(
        arch=args.arch,
        hidden_dim=args.hidden_dim,
        channels=args.channels,
        num_blocks=args.num_blocks,
        pt_path=pt_path_obj,
        onnx_path=onnx_path_obj,
    )
