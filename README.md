# PuyotanAI (Architectural Refactor)

PuyotanAIは、極限まで最適化されたC++ネイティブエンジンと強化学習（PPO/GAE）を融合させた、超高速なぷよぷよAI訓練パイプラインです。

## 🚀 特徴
- **極限のパフォーマンス**: 10M+ FPSを超えるシミュレーション速度（AMD 3020e等の省電力環境でも動作）。
- **戦略的イニシアチブ**: 毎ステップ24通りの仮想ドロップをシミュレートし、相手の隙を突く「攻め」や「プレッシャー」の概念を学習。
- **C++ネイティブ報酬エンジン**: 全ての重みパラメータをJSONで外部化し、高速性を維持したまま柔軟なチューニングが可能。
- **完全 branchless ロジック**: BitboardとSIMDを駆使した、CPUパイプラインに優しい高速設計。

## 📂 プロジェクト構造
- `native/`: C++ ゲームエンジン、RL環境、高速報酬計算ロジック。
    - `native/resources/reward_default.json`: 中央管理された報酬パラメータ。
- `training/`: Python による学習（PPO）およびモデル管理。
- `orchestrator/`: 学習の実行・制御・視覚化。
    - `solo_training.py`: スコア最大化を目指す基礎訓練。
    - `selfplay.py`: 過去の自分と戦う実践訓練。
- `gui/`: Webベースの可視化・対戦インターフェース。

## 🛠 使い方

### 1. ビルド
`native/` ディレクトリにある `build.bat` を使用してビルドするのが最も簡単かつ高速です。
```powershell
cd native
.\build.bat
```
このスクリプトは、AVX2命令や高レベルな最適化オプション（/O2, /GL, /fp:fast）を自動的に適用します。

### 2. 報酬の調整
`native/resources/reward_default.json` を編集することで、AIの「性格」を変更できます。再コンパイルは不要です。

### 3. 学習の開始
```powershell
python orchestrator/solo_training.py  # 基礎訓練
python orchestrator/selfplay.py       # 大会レベルへの強化
```

## ⚖️ ライセンス
- ソフトウェア本体: MIT License
- 外部ライブラリ (`json.hpp`): [nlohmann/json](https://github.com/nlohmann/json) (MIT License) - Vendored for portability.
