"""
報酬計算モジュール（単一責務）

報酬設計の方針:
  - 主軸: 勝利/敗北（±10.0）
  - 中間報酬: スコア増加（おじゃまを送る力）、おじゃまを受ける量
  - 観測: 自分と相手の両フィールドを考慮可能（obs形状 [2, 5, 6, 13]）
"""
import puyotan_native as p


def calculate_reward(
    prev_p1_score: int,
    curr_p1_score: int,
    prev_p1_ojama: int,
    curr_p1_ojama: int,
    match_status,
) -> float:
    """
    1ステップ分の報酬を計算する。

    Args:
        prev_p1_score:  前ステップのP1スコア
        curr_p1_score:  現ステップのP1スコア
        prev_p1_ojama:  前ステップのP1が受けたおじゃま量
        curr_p1_ojama:  現ステップのP1が受けたおじゃま量
        match_status:   現在の試合状態

    Returns:
        float: この1ステップの報酬
    """
    reward = 0.0

    # おじゃまを送る力（スコア増加）に比例してプラス
    score_delta = curr_p1_score - prev_p1_score
    reward += score_delta * 0.002

    # おじゃまを受けることにペナルティ
    ojama_delta = curr_p1_ojama - prev_p1_ojama
    reward -= ojama_delta * 0.001

    # 終了報酬（主軸）
    if match_status == p.MatchStatus.WIN_P1:
        reward += 10.0
    elif match_status == p.MatchStatus.WIN_P2:
        reward -= 10.0

    return float(reward)
