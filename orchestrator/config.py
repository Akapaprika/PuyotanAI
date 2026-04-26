from dataclasses import dataclass, field, replace
from typing import Dict

@dataclass
class TrainingProfile:
    """アーキテクチャごとの学習プロファイル定義"""

    # --- 計算リソース (データの収集量) ---
    # 並列で走るゲーム環境の数。多ければ多いほど多様なデータが一度に手に入るがCPUを食う。
    NUM_ENVS: int
    # 1回の学習（1イテレーション）を行うまでに、各環境が何手（ステップ）進むか。
    # NUM_ENVS * STEPS_PER_ITER = 1回の更新に使われる総データ数（Batch Size）。
    STEPS_PER_ITER: int

    # --- ログ・保存間隔 ---
    LOG_INTERVAL: int
    SAVE_INTERVAL: int
    TOTAL_ITERS: int = 1000

    # --- モデル構造 ---
    # MLP等の隠れ層のノード数。大きいほど表現力（賢さ）が上がるが計算が重くなる。
    HIDDEN_DIM: int = 128

    # --- PPO ハイパーパラメータ (学習の質に直結) ---
    # 学習率（Learning Rate）: 1回の学習でどれくらい大きくパラメータを更新するか。
    # 大きすぎると学習が崩壊し、小さすぎると一向に賢くならない。
    LEARNING_RATE: float = 3e-4
    
    # エポック数（Epochs）: 集めたデータを何周繰り返し学習に使うか。
    # PPOは「集めたデータを少しずつ複数回学ぶ」ことで効率を上げる。
    NUM_EPOCHS: int      = 4
    
    # ミニバッチサイズ（Minibatch）: 1回の勾配降下（パラメータ更新）に使うブロックの大きさ。
    # 総データ数（Num_Envs * Steps）をこれに分割して学習する。並列計算効率に影響する。
    MINIBATCH: int       = 4096
    
    # 割引率（Gamma, γ）: 将来の報酬をどれくらい割引いて現在の価値とするか (0.0〜1.0)
    # 0.99 の場合、100手先の連鎖報酬もある程度考慮して現在の積み手を選ぶようになる。
    # 連鎖を作るには中長期の見通しが必要なため、短期報酬寄り(0.1)ではなく標準的な 0.99 を採用。
    GAE_GAMMA: float     = 0.99
    
    # 汎化アドバンテージ推定（Lambda, λ）: どれくらい遠い未来の予測を信用するか (0.0〜1.0)
    # 値が高いほど実際の報酬に引っ張られ（バリアンス高）、低いほど予測に頼る（バイアス高）。
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
    STEPS_PER_ITER = 64,
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
    STEPS_PER_ITER = 64,
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

    # グローバルな設定テンプレートを直接変更しないよう、呼び出しごとに複製を返す。
    # （以前は self-play 呼び出しで SNAPSHOT_INTERVAL を変更すると、
    #   同一プロセス内の後続呼び出しにも副作用が残る可能性があった）
    cfg = replace(cfg)

    # セルフプレイ時のみスナップショットを有効化
    if is_selfplay:
        cfg.SNAPSHOT_INTERVAL = 10

    return cfg
