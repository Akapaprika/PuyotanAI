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
│   ├── train.py              # 訓練プログラム（ソロ／セルフプレイ統合）
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

トレーニングは `orchestrator/train.py` に一本化されています。

| 引数 | 選択肢 | デフォルト | 説明 |
|---|---|---|---|
| `--mode` | `solo` / `selfplay` | `solo` | 訓練モード |
| `--arch` | `mlp` / `light_mlp` / `cnn` / `resnet` | `mlp` | ネットワーク構造 |
| `--config` | JSONファイル名 | mode依存 ※ | 報酬パラメータ設定ファイル |
| `--lr` | float | config依存 | 学習率 |
| `--envs` | int | config依存 | 並列環境数 |
| `--steps` | int | config依存 | 1イテレーション当たりの収集ステップ数 |
| `--channels` | int | config依存 | CNN / ResNet のチャンネル数 |
| `--num_blocks` | int | config依存 | ResNet の残差ブロック数 |
| `--load` | .pt ファイルパス | ─ | 指定チェックポイントから再開 |
| `--fresh` | flag | ─ | 既存チェックポイントを無視して最初から学習 |

> ※ `--mode solo` → `reward_solo.json`、`--mode selfplay` → `reward_match.json`

---

#### 【モード1】ソロトレーニング — まずはここから

スコア最大化・連鎖構築の「基礎体力」を鍛えます。P2はミラー（鏡写し）行動を行い、お邪魔ぷよは相殺されます。

```powershell
python -m orchestrator.train                   # デフォルト（MLP・ソロ）
python -m orchestrator.train --arch light_mlp  # 超軽量（低スペックPC向け）
python -m orchestrator.train --arch cnn        # 盤面形状を画像として認識
python -m orchestrator.train --arch resnet     # より深い盤面理解（高スペック推奨）
```

---

#### 【モード2】セルフプレイ — 対戦AIへの強化

「過去の自分」をフリーズして対戦相手とし、実際の対戦に勝つための戦術を学習します。
ソロ訓練のチェックポイントが存在する場合、そこから自動的に引き継いで学習を開始します。

```powershell
python -m orchestrator.train --mode selfplay               # デフォルト（MLP）
python -m orchestrator.train --mode selfplay --arch cnn
python -m orchestrator.train --mode selfplay --arch resnet
```

---

#### 【オプション】チェックポイントと再開

```powershell
# 指定のチェックポイントから再開する
python -m orchestrator.train --load models/mlp/puyotan_solo_latest.pt

# 既存のチェックポイントを無視して最初から訓練する
python -m orchestrator.train --fresh
python -m orchestrator.train --mode selfplay --fresh
```

---

### 🔍 探索AI（ビームサーチ）

PPOによる直感型AIに加え、C++エンジンによる高速なシミュレーションを活かした**ビームサーチAI**を利用可能です。

- **`BeamSearchAgent`**: ニューラルネットワーク不要。次の3手先までの全配置パターンをシミュレーションして最善手を選びます。
- **`HybridBeamOnnxAgent`**: 平時はビームサーチで大連鎖を構築し、お邪魔ぷよ着弾時は学習済みPPO（ONNX）に切り替えて防御するハイブリッドAIです。

---

### ⚙️ ビームサーチ設定のチューニング (`gui/beam_config.json`)

ビームサーチAIの探索動作や評価関数の重みは、`gui/beam_config.json` で一元管理されています。この設定はC++側でネイティブに読み込まれます（**再コンパイル不要**）。

#### 📊 基本設定項目
- **`beam_width`** (int): 各深さで保持する最善の盤面候補数。値を大きくすると強くなりますが、処理時間が増加します（推奨: 300〜1000）。
- **`look_ahead`** (int): 先読みするツモ数。標準では3手先（ネクネク）まで先読み可能です。

#### 🧠 評価関数の重み (`eval_weights`)
盤面の良さを評価するための各種パラメータです：
- **`potential_score_scale`** (float): 将来構築可能な連鎖ポテンシャルスコアへの乗数。高いほど将来の大連鎖を優先します。
- **`connectivity_bonus`** (float): 2個以上連結している色ぷよ1個あたりのボーナス。
- **`isolated_penalty`** (float): 隣接ぷよが0の孤立ぷよ1個あたりのペナルティ（負の値）。
- **`buried_penalty`** (float): おじゃまぷよの下に埋もれて消せなくなった色ぷよ1個あたりのペナルティ（負の値）。
- **`height_variance_penalty`** (float): 各列の高さの分散（凹凸）に掛けるペナルティ（負の値）。平坦な盤面を維持しやすくなります。
- **`death_col_penalty`** (float): 窒息列（3列目）の高さ1行あたりのペナルティ（負の値）。
- **`chain_bonus_per_step`** (float): その手で発火した連鎖に対する基本ボーナス。
- **`chain_power`** (float): 連鎖ボーナスの指数（2.0の場合、`連鎖数 ^ 2.0` に比例したボーナスが加算されます）。
- **`use_fast_potential`** (bool): `true` にすると、ポテンシャルの計算に高速な簡易アルゴリズムを使用し、処理速度を向上させます。

#### 🔄 動的プロファイル (`profiles`)
ゲームの状況やモードに応じて、上記の評価重みに差分パッチを適用する機能です：
- **`deep_search`**: 先読み深さが4手以上の場合に自動適用。将来が見通せるため、平坦化ペナルティなどを緩めて大連鎖構築に特化させます。
- **`vs_mode`** / **`solo_mode`**: 対戦モード / ソロモード（とことん）に応じて自動適用。対戦時はおじゃまペナルティを厳格化し、ソロ時はおじゃま無視で効率を最大化します。
- **`stagnated`**: 盤面が硬直し、過去3手で期待スコアが伸びていない停滞状態を検出した時に自動適用。即時発火を最優先にして盤面をリセット（お掃除）します。

---

### Step 3: 報酬のチューニング

`native/resources/` 内のJSONを編集するだけで、AIの「性格」を変更できます。**再コンパイル不要**。

- `reward_solo.json` — 連鎖構築・スコア最大化重視のパラメータ
- `reward_match.json` — 対戦向け・おじゃまぷよ攻撃・戦略的イニシアチブのパラメータ

---

### Step 4: GUI対戦・観戦（Match Viewer）の起動

ビジュアル環境での対戦や、AIどうしの対戦（または人間対AI、人間どうしの対戦）を起動します。

```powershell
python -m gui.main
```

起動後、セットアップ画面で各プレイヤーのエージェント（Human / AI (ONNX) / Beam Search / Hybrid (Beam+ONNX) / Empty）を選択して「Start Match」をクリックしてください。
AIやHybridを使用する場合は、事前にONNXモデルを書き出しておく必要があります：
```powershell
python -m orchestrator.export_onnx --arch mlp
```

---

## 📊 推奨学習フロー

```
[ソロ訓練] train.py --mode solo       ─→  連鎖を安定して組めるようになる
        ↓ (チェックポイントを引継ぎ)
[セルフプレイ] train.py --mode selfplay ─→  対戦相手への攻撃・防御を習得
```

---

## ⚖️ ライセンス

- ソフトウェア本体: MIT License
- 外部ライブラリ (`json.hpp`): [nlohmann/json](https://github.com/nlohmann/json) (MIT License) - Vendored for portability.
