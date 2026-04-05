from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class TrainingProfile:
    """アーキテクチャごとの学習プロファイル定義"""
    # 計算リソース
    NUM_ENVS: int
    STEPS_PER_ITER: int
    
    # ログ・保存間隔
    LOG_INTERVAL: int
    SAVE_INTERVAL: int
    TOTAL_ITERS: int = 1000

    # モデル構造
    HIDDEN_DIM: int = 128

    # PPO ハイパーパラメータ
    LEARNING_RATE: float = 3e-4
    NUM_EPOCHS: int      = 4
    MINIBATCH: int       = 4096
    GAE_GAMMA: float     = 0.99
    GAE_LAMBDA: float    = 0.95

    # セルフプレイ専用（デフォルト無効）
    SNAPSHOT_INTERVAL: int = 0

# ---------------------------------------------------------------------------
# 1. MLP プロファイル (爆速・大量データ向け)
# ---------------------------------------------------------------------------
MLP_CONFIG = TrainingProfile(
    NUM_ENVS=256,
    STEPS_PER_ITER=128,
    LOG_INTERVAL=1,
    SAVE_INTERVAL=50,      # 速いので保存は控えめ
    TOTAL_ITERS=2000,      # たくさん回せる
    MINIBATCH=8192         # 大量データを一気に処理
)

# ---------------------------------------------------------------------------
# 2. ResNet プロファイル (重厚・慎重・高頻度保存向け)
# ---------------------------------------------------------------------------
RESNET_CONFIG = TrainingProfile(
    NUM_ENVS=64,           # 負荷を抑える
    STEPS_PER_ITER=32,     # 1ステップを短く
    LOG_INTERVAL=1,
    SAVE_INTERVAL=10,      # 遅いのでこまめに保存して進捗を守る
    TOTAL_ITERS=1000,
    MINIBATCH=1024         # メモリと計算量を節約
)

# ---------------------------------------------------------------------------
# 3. CNN プロファイル (中間)
# ---------------------------------------------------------------------------
CNN_CONFIG = TrainingProfile(
    NUM_ENVS=128,
    STEPS_PER_ITER=64,
    LOG_INTERVAL=1,
    SAVE_INTERVAL=20,
    TOTAL_ITERS=1500,
    MINIBATCH=2048
)

def get_config(arch: str, is_selfplay: bool = False) -> TrainingProfile:
    """指定されたアーキテクチャとモードに最適なコンフィグを返却する"""
    arch = arch.lower()
    if arch == "resnet":
        cfg = RESNET_CONFIG
    elif arch == "cnn":
        cfg = CNN_CONFIG
    else:
        cfg = MLP_CONFIG # デフォルトは MLP

    # セルフプレイモードの場合の共通調整
    if is_selfplay:
        # セルフプレイ時はスナップショットが必要
        cfg.SNAPSHOT_INTERVAL = 10 
        
    return cfg
