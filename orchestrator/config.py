from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TrainingProfile:
    """アーキテクチャごとの学習プロファイル定義"""

    # --- 計算リソース ---
    NUM_ENVS: int
    STEPS_PER_ITER: int

    # --- ログ・保存間隔 ---
    LOG_INTERVAL: int
    SAVE_INTERVAL: int
    TOTAL_ITERS: int = 1000

    # --- モデル構造 ---
    HIDDEN_DIM: int = 128

    # --- PPO ハイパーパラメータ ---
    LEARNING_RATE: float = 3e-4
    NUM_EPOCHS: int      = 4
    MINIBATCH: int       = 4096
    GAE_GAMMA: float     = 0.99
    GAE_LAMBDA: float    = 0.95

    # --- アーキテクチャ固有パラメータ (C++ 側に辞書で渡す) ---
    # MLP は不要。CNN/ResNet で有効。
    # クラウド/GPUで学習する場合はここを拡張する。
    ARCH_PARAMS: Dict[str, int] = field(default_factory=dict)

    # --- セルフプレイ専用（デフォルト無効） ---
    SNAPSHOT_INTERVAL: int = 0

# ---------------------------------------------------------------------------
# 1. MLP プロファイル
#    並列環境数を最大化できる爆速アーキテクチャ向け
# ---------------------------------------------------------------------------
MLP_CONFIG = TrainingProfile(
    NUM_ENVS       = 256,
    STEPS_PER_ITER = 128,
    LOG_INTERVAL   = 1,
    SAVE_INTERVAL  = 50,
    TOTAL_ITERS    = 2000,
    MINIBATCH      = 8192,
    ARCH_PARAMS    = {},      # MLP は構造変更不要
)

# ---------------------------------------------------------------------------
# 2. CNN プロファイル
#    GAP 最適化済み。channels=64 は 2コア CPU でも快適に動作する。
#    クラウドで使うなら channels=128 以上に変更するだけ。
# ---------------------------------------------------------------------------
CNN_CONFIG = TrainingProfile(
    NUM_ENVS       = 128,
    STEPS_PER_ITER = 64,
    LOG_INTERVAL   = 1,
    SAVE_INTERVAL  = 20,
    TOTAL_ITERS    = 1500,
    MINIBATCH      = 2048,
    ARCH_PARAMS    = {"channels": 64},
)

# ---------------------------------------------------------------------------
# 3. ResNet プロファイル
#    GAP 最適化済み軽量設定（2コアCPU 向け）。
#    GPU/クラウド向けには channels=64, num_blocks=4 以上を推奨。
# ---------------------------------------------------------------------------
RESNET_CONFIG = TrainingProfile(
    NUM_ENVS       = 64,
    STEPS_PER_ITER = 32,
    LOG_INTERVAL   = 1,
    SAVE_INTERVAL  = 10,
    TOTAL_ITERS    = 1000,
    MINIBATCH      = 1024,
    ARCH_PARAMS    = {
        "channels":   32,   # 軽量: 2コア CPU 向け (GAP後 fc_in=32)
        "num_blocks": 2,    # 軽量: ResNetブロック数
        # ↓ GPU/クラウド移行時の設定例 (コメント解除するだけ)
        # "channels":   64,
        # "num_blocks": 4,
    },
)

def get_config(arch: str, is_selfplay: bool = False) -> TrainingProfile:
    """指定アーキテクチャとモードに最適なプロファイルを返す。"""
    _arch = arch.lower()
    if _arch == "resnet":
        cfg = RESNET_CONFIG
    elif _arch == "cnn":
        cfg = CNN_CONFIG
    else:
        cfg = MLP_CONFIG  # デフォルトは MLP

    # セルフプレイ時のみスナップショットを有効化
    if is_selfplay:
        cfg.SNAPSHOT_INTERVAL = 10

    return cfg
