# PuyotanAI 報酬パラメータ仕様書

このドキュメントでは、`reward_solo.json` や `reward_match.json` で設定可能な各パラメータの意味と、C++ エンジン内での計算式について説明します。

---

## 報酬計算の基本原則
1ステップ（1アクション）ごとに報酬が計算されます。
計算式は基本的に **`報酬 = Σ(各メトリクスの値 × 設定した重み)`** です。

---

## 1. 試合結果 (match)
エピソード（試合）が終了した瞬間に一度だけ加算されます。

| パラメータ | 説明 |
| :--- | :--- |
| `win` | 勝利時に加算される報酬。（例: 25.0） |
| `loss` | 敗北時に加算的される報酬。（例: -50.0） |
| `draw` | 引き分け時に加算される報酬。 |

---

## 2. ターン制約 (turn)
AIの「速さ」を制御します。

| パラメータ | 説明 | 計算式 |
| :--- | :--- | :--- |
| `step_penalty` | 1ステップ（1手）ごとに加算。 | `+ step_penalty` |

> [!TIP]
> 負の値を設定することで、AIは「無駄な手を打たず、最短で勝利する」ことを学習します。

---

## 3. パフォーマンス (performance)
連鎖やスコアなど、直接的な「上手さ」を評価します。

| パラメータ | 説明 | 計算式 |
| :--- | :--- | :--- |
| `score_scale` | 1手で得たゲームスコア（点数）。 | `delta_score * score_scale` |
| `chain_bonus_scale` | 連鎖の大きさに対するボーナス。 | `(連鎖数^2) * chain_bonus_scale` |

> [!NOTE]
> `chain_bonus_scale` は連鎖数の2乗に比例するため、大連鎖を組む動機付けが強力になります。

---

## 4. 盤面状態 (board)
自分自身の盤面構成を評価します。

| パラメータ | 説明 | 計算式 |
| :--- | :--- | :--- |
| `puyo_count_penalty` | 盤面にある全ぷよ数へのペナルティ。 | `ぷよ数 * penalty` |
| `connectivity_bonus` | ぷよの連結。組みやすさの指標。 | `連結スコア * bonus` |
| `isolated_puyo_penalty` | どこにも繋がっていない「ゴミぷよ」の数。 | `個数 * penalty` |
| `death_col_height_penalty` | 3列目（窒息列）の高さ。 | `高さ * penalty` |
| `color_diversity_reward` | 盤面にある色の種類数。 | `色数 * reward` |
| `buried_puyo_penalty` | お邪魔の下に埋まった自分のぷよ。 | `埋まり数 * penalty` |
| `ojama_drop_penalty` | その手で降ってきたお邪魔ぷよの数。 | `降下数 * penalty` |
| `potential_chain_bonus_scale` | **「あと1手で最大何連鎖できるか」** | `(最大可能連鎖^2) * scale` |

---

## 5. 対戦相手との関係 (opponent)
「相手を邪魔する」「相手より有利に立つ」ことを評価します。

| パラメータ | 説明 | 計算式 |
| :--- | :--- | :--- |
| `field_pressure_reward` | 相手の盤面にあるぷよの総数。 | `相手のぷよ数 * reward` |
| `connectivity_penalty` | 相手のぷよの連結状態へのペナルティ。 | `相手の連結 * penalty` |
| `ojama_diff_scale` | スコア差（送り込んだ量の差）。 | `(自スコア - 相手スコア) * scale` |
| `initiative_bonus` | **戦略的イニシアチブ** | 自分が発火可能＆相手が不可なら「固定値」加算 |

> [!IMPORTANT]
> **initiative_bonus** は、対人戦において非常に重要です。
> 「自分はいつでも連鎖を撃てるが、相手は準備ができていない」という有利な状況を高く評価します。

---

## 設定のヒント
- **ソロトレーニング用**: `connectivity_bonus` や `potential_chain_bonus_scale` を高めにして「形をきれいに組むこと」を優先させます。
- **対人戦用**: `initiative_bonus` や `ojama_diff_scale` を高めにして、相手の隙を突く戦術を重視させます。
