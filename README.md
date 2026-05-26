# PuyotanAI

PuyotanAIは、極限まで最適化されたC++ネイティブエンジンと強化学習（PPO/GAE）を融合させた、超高速なぷよたんAI訓練パイプラインです。

ぷよぷよとほとんど同じルールですが、ゲームの進め方だけ違います。
RPGのコマンド選択のように両プレーヤーがツモ操作をするまでゲームが進行しません。リアルタイム性を排除することでじっくり考えて、相手の盤面を把握して行動選択するのが重要なゲームです。

注意点として、ぷよの消滅・設置・落下に対して一定数のターンが消費されます（硬直時間）。コード内ではこのゲームにおけるターン制のことを「フレーム」としています。いわゆるフレームレートとは全く異なる概念です。

---

## 🚀 特徴

- **極限のパフォーマンス**: 10M+ FPSを超えるシミュレーション速度（AMD 3020e等の省電力環境でも動作）。
- **C++ネイティブ RL**: PPO（ロールアウト収集・GAE計算・勾配更新）をすべてLibTorchでネイティブ実行。Pythonのオーバーヘッドを排除。
- **JSONパラメータ設計**: 報酬の全重みをJSONで外部化。再コンパイル不要でリアルタイムチューニング可能。
- **完全 branchless ロジック**: BitboardとSIMDを駆使した、CPUパイプラインに優しい高速設計。

---

## 📂 プロジェクト構造

```
PuyotanAI/
├── native/                   # C++ ゲームエンジン・RL環境・報酬計算
│   ├── resources/
│   │   ├── reward_solo.json  # ソロ訓練用報酬パラメータ
│   │   └── reward_match.json # 対戦訓練用報酬パラメータ
│   └── dist/                 # ビルド済みネイティブモジュール (.pyd)
├── orchestrator/             # 学習の実行・制御スクリプト
│   ├── solo_training.py      # 【モード1】ソロ訓練（スコア最大化・連鎖構築）
│   ├── selfplay.py           # 【モード2】セルフプレイ訓練（対戦AI育成）
│   └── config.py             # ハイパーパラメータ一元管理
├── models/                   # 学習済みモデルのチェックポイント
│   ├── mlp/
│   │   ├── puyotan_solo_latest.pt   # ソロ訓練の最新モデル
│   │   └── puyotan_latest.pt        # セルフプレイの最新モデル
│   └── cnn/ resnet/ light_mlp/      # アーキテクチャ別
├── gui/                      # 対戦・観戦インターフェース
└── memo.md                   # 開発・技術詳細メモ
```

---

## 🛠 使い方

### Step 1: ビルド

```powershell
cd native
.\build.bat
```

AVX2命令・高レベル最適化（/O2, /GL, /fp:fast）が自動適用されます。

---

### Step 2: 学習モードの選択

#### 【モード1】ソロトレーニング — まずはここから

スコア最大化・連鎖構築の「基礎体力」を鍛えます。対戦相手は存在しません（P2はランダム行動）。

```powershell
# デフォルト（MLP・256並列環境）
python -m orchestrator.solo_training

# アーキテクチャを指定
python -m orchestrator.solo_training --arch mlp        # 高速・省メモリ
python -m orchestrator.solo_training --arch light_mlp  # 超軽量（低スペックPC向け）
python -m orchestrator.solo_training --arch cnn        # 盤面形状を画像として認識
python -m orchestrator.solo_training --arch resnet     # より深い盤面理解（高スペック推奨）

# 報酬設定ファイルを指定（デフォルト: reward_solo.json）
python -m orchestrator.solo_training --config reward_solo.json
```

| オプション | 選択肢 | デフォルト | 説明 |
|---|---|---|---|
| `--arch` | `mlp` / `light_mlp` / `cnn` / `resnet` | `mlp` | ネットワーク構造 |
| `--config` | `reward_solo.json` など | `reward_solo.json` | 報酬パラメータJSON |

---

#### 【モード2】セルフプレイ — 対戦AIへの強化

「過去の自分」をフリーズして対戦相手とし、実際の対戦に勝つための戦術を学習します。
ソロ訓練でチェックポイント（`puyotan_latest.pt`）が存在する場合、そこから引き継いで学習します。

```powershell
# デフォルト（MLP）
python -m orchestrator.selfplay

# アーキテクチャを指定
python -m orchestrator.selfplay --arch mlp
python -m orchestrator.selfplay --arch cnn

# 報酬設定を指定（デフォルト: reward_match.json）
python -m orchestrator.selfplay --reward reward_match.json
```

| オプション | 選択肢 | デフォルト | 説明 |
|---|---|---|---|
| `--arch` | `mlp` / `light_mlp` / `cnn` / `resnet` | `mlp` | ネットワーク構造 |
| `--reward` | `reward_match.json` など | `reward_match.json` | 報酬パラメータJSON |

---

### Step 3: 報酬のチューニング

`native/resources/` 内のJSONを編集するだけで、AIの「性格」を変更できます。**再コンパイル不要**。

- `reward_solo.json` — 連鎖構築・スコア最大化重視のパラメータ
- `reward_match.json` — 対戦向け・おじゃまぷよ攻撃・戦略的イニシアチブのパラメータ

---

## 📊 推奨学習フロー

```
[ソロ訓練] solo_training.py   ─→  連鎖を安定して組めるようになる
        ↓ (チェックポイントを引継ぎ)
[セルフプレイ] selfplay.py    ─→  対戦相手への攻撃・防御を習得
```

---

## ⚖️ ライセンス

- ソフトウェア本体: MIT License
- 外部ライブラリ (`json.hpp`): [nlohmann/json](https://github.com/nlohmann/json) (MIT License) - Vendored for portability.
