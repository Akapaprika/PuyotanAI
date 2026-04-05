# PuyotanAI 報酬パラメータ仕様書 (2026/04/05 改訂)

このドキュメントでは、`reward_solo.json` や `reward_match.json` で設定可能な全パラメータの意味と、C++ エンジン内での計算ロジックについて詳細に説明します。

---

## 1. 試合結果 (match)
エピソード（試合）終了時に1度だけ加算されます。

| パラメータ | 説明 |
| :--- | :--- |
| `win` | 勝利時の報酬。 |
| `loss` | 敗北時の報酬（通常は負の値）。 |
| `draw` | 引き分け時の報酬。 |

---

## 2. ターン制約 (turn)
AIの「効率性」を制御します。

| パラメータ | 説明 |
| :--- | :--- |
| `step_penalty` | 1手ごとに加算される値。負の値にすることで、最短勝利を促します。 |

---

## 3. パフォーマンス (performance)
連鎖や消去アクションをより細かく評価します。

| パラメータ | 説明 | 計算式 |
| :--- | :--- | :--- |
| `score_scale` | 獲得スコア。 | `delta_score * score_scale` |
| `chain_scale` | 連鎖ボーナスの倍率。 | `(連鎖数 ^ chain_power) * chain_scale` |
| `chain_power` | 連鎖数の累乗指数。2.0なら2乗、1.0なら線形評価。 | (上記参照) |
| `min_chain_threshold` | この値未満の連鎖には `premature_chain_penalty` が適用されます。 | - |
| `premature_chain_penalty` | 小さすぎる連鎖（暴発など）に対する重み。 | `連鎖数 * penalty` |
| `all_clear_bonus` | 全消し（All Clear）達成時のボーナス。 | 固定値加算 |
| `erasure_count_scale` | 消したぷよの総数（連結加算等を除く純粋な個数）。 | `消去数 * scale` |
| `ojama_sent_scale` | 相手に送り込んだお邪魔ぷよの数。 | `送り数 * scale` |

---

## 4. 盤面状態 (board)
自分自身の盤面構成の「美しさ」や「危険度」を評価します。

| パラメータ | 説明 |
| :--- | :--- |
| `puyo_count_penalty` | 盤面にある全ぷよ数。基本は負の値を設定し、無駄なぷよを減らします。 |
| `connectivity_bonus` | 同じ色のぷよが隣接している状態。組みやすさの指標。 |
| `isolated_puyo_penalty` | 周囲に同色がない「ゴミぷよ」の数へのペナルティ。 |
| `near_group_bonus` | 「あと1つで消える（3連結）」状態の数。発火の準備度。 |
| `height_variance_penalty` | 各列の高さの分散。平らでない盤面へのペナルティ。 |
| `death_col_height_penalty` | 敗北判定列（通常3列目）の高さ。 |
| `color_diversity_reward` | 盤面にある色の種類数。多色保持を促す場合に。 |
| `buried_puyo_penalty` | お邪魔ぷよの下に埋まった自分の色ぷよの数。 |
| `ojama_drop_penalty` | その手で降ってきたお邪魔ぷよの数。 |
| `pending_ojama_penalty` | 予告トレイにある（まだ降っていない）お邪魔ぷよの総数。 |
| `potential_chain_bonus_scale` | **[一時停止中]** 「あと1手で最大何連鎖できるか」の2乗に比例。性能向上のため現在は0固定。 |

---

## 5. 対戦相手との関係 (opponent)
「相手への圧力」と「状況的有利」を評価します。

| パラメータ | 説明 |
| :--- | :--- |
| `field_pressure_reward` | 相手の盤面にある「ぷよの総数」が多いほど報酬。 |
| `connectivity_penalty` | 相手の盤面で「ぷよが繋がっている」ほどペナルティ。相手の形を崩す動機。 |
| `ojama_diff_scale` | 送り込んだお邪魔ぷよの「差分」（自スコア - 敵スコア）に基づく。 |
| `initiative_bonus` | **[一時停止中]** 戦略的イニシアチブ。自分が発火可能で、相手が不可な状態への加点。 |

---

## 設定のヒント
*   **探索なしPPO**: `potential_chain_bonus_scale` や `near_group_bonus` を厚くすることで、先読みしなくても「良い形」を維持できるようになります。
*   **対人戦**: `pending_ojama_penalty` や `ojama_diff_scale` を調整することで、相手が打ってくる前に察知して対応する動きが生まれます。
