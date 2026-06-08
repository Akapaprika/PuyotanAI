"""
gui/agents.py

Player Agent (Strategy) pattern.
Each agent is responsible for providing a p.Action when the engine marks
the player as needing a decision (decision_mask bit is set).

  HumanPlayerAgent  — waits for keyboard/button input buffered in pres state
  AIPlayerAgent     — runs ONNX inference via puyotan_native.OnnxPolicy
  EmptyPlayerAgent  — immediately PASSes, creating an uncontested 1P side
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import puyotan_native as p


class BasePlayerAgent(ABC):
    """Abstract interface for a player controller."""

    @abstractmethod
    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        """
        Return an action for the current decision point, or None if the
        agent is still waiting (e.g. human hasn't pressed confirm yet).

        Parameters
        ----------
        match      : GameModel — thin wrapper around PuyotanMatch
        player_id  : int       — 0 or 1
        pres       : PlayerPresentationState — UI-layer state for this player
        """


# ---------------------------------------------------------------------------
# Human
# ---------------------------------------------------------------------------
class HumanPlayerAgent(BasePlayerAgent):
    """
    Returns a PUT action when the user has confirmed their placement
    (pres.confirmed == True), otherwise returns None to keep waiting.
    """

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        if pres.confirmed:
            return p.Action(p.ActionType.PUT, pres.x, pres.rotation)
        return None


# ---------------------------------------------------------------------------
# Base AI Agent (ONNX Wrapper)
# ---------------------------------------------------------------------------
class BaseAIAgent(BasePlayerAgent):
    """
    Base class for AI agents running ONNX policy inference.
    Handles loading, reloading, and executing the ONNX policy.
    """

    # Observation size constants (must match ObservationBuilder in C++)
    OBS_SIZE = 2 * 5 * 6 * 14  # kBytesPerObservation

    # All 22 PUT actions, in exactly the same order as the C++ training loop.
    _VALID_ACTIONS: list[p.Action] = [
        p.get_rl_action(i) for i in range(p.kNumRLActions)
    ]

    def __init__(self, model_path: str) -> None:
        self.policy = p.OnnxPolicy(model_path, use_cpu=True)
        self._obs_buf = np.zeros(self.OBS_SIZE, dtype=np.uint8)

    def reload(self, model_path: str) -> None:
        """Hot-swap the underlying ONNX model."""
        self.policy = p.OnnxPolicy(model_path, use_cpu=True)

    def _infer_action(self, obs_flat: np.ndarray) -> p.Action:
        """Runs the ONNX model inference on a flat observation tensor."""
        obs_batch = obs_flat.reshape(1, -1)
        action_indices = self.policy.infer(obs_batch)  # list of int
        idx = int(action_indices[0])
        idx = max(0, min(idx, len(self._VALID_ACTIONS) - 1))
        return self._VALID_ACTIONS[idx]

    def _flip_observation_if_needed(self, obs_flat: np.ndarray, player_id: int) -> np.ndarray:
        """Flips the observation perspective if the player is P2 (index 1)."""
        if player_id == 1:
            half = self.OBS_SIZE // 2
            return np.concatenate([obs_flat[half:], obs_flat[:half]])
        return obs_flat


# ---------------------------------------------------------------------------
# AI (ONNX)
# ---------------------------------------------------------------------------
class AIPlayerAgent(BaseAIAgent):
    """
    Runs ONNX inference on every decision point.
    The action is chosen immediately — no human latency.
    """

    def get_action(self, match, player_id: int, pres) -> p.Action:
        # Build observation tensor from the engine state
        obs_flat = p.build_observation(match.match)  # shape (OBS_SIZE,)
        obs_flat = self._flip_observation_if_needed(obs_flat, player_id)
        return self._infer_action(obs_flat)


# ---------------------------------------------------------------------------
# Empty (Solo / Pass-through)
# ---------------------------------------------------------------------------
class EmptyPlayerAgent(BasePlayerAgent):
    """
    Mirrors the other player's PUT action exactly (solo / tokoton mode).

    Because the tsumo queue is shared, copying the same (x, rotation) each
    turn keeps both fields in perfect sync.  Any ojama sent by the human's
    chains will be countered by identical chains on the mirrored side, so
    it never accumulates on either board.

    Returns None (wait) until the other player has committed their PUT so
    that we always read a valid action.
    """

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        other_id = 1 - player_id
        other_state = match.get_player_state(other_id)
        other_action = other_state.current_action.action
        if other_action.type == p.ActionType.PUT:
            # Mirror exactly — same column and rotation
            return p.Action(p.ActionType.PUT, other_action.x, other_action.rotation)
        # Other player hasn't confirmed yet — keep waiting
        return None


# ---------------------------------------------------------------------------
# Beam Search AI
# -----------------------------------------------------------# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------
def _adjust_eval_weights(cfg, is_solo: bool, is_stagnated: bool) -> None:
    w = cfg.eval_weights
    
    # 1. 探索の深さ (look_ahead) に応じた調整 (総当たりによる影響度の動的変化)
    if cfg.look_ahead >= 4:
        # 深層探索の時：
        # 将来の連鎖が総当たり（Expectimax）で見通せるため、
        # 目先の連鎖をすぐに打つボーナスと、盤面を無理やり平坦にするペナルティを弱める。
        w.height_variance_penalty = -0.05  # デフォルト -0.3 から大幅緩和（大連鎖用の起伏を許容）
        w.chain_bonus_per_step = 0.5       # デフォルト 2.0 から大幅緩和（大連鎖完成まで発火を我慢）
        w.connectivity_bonus = 0.5         # デフォルト 0.4 から少し引き上げ
    else:
        # 浅層探索（D<=3）の時：
        # 将来が見えにくいため、平坦さの保険や即時発火ボーナスを維持。
        w.height_variance_penalty = -0.3
        w.chain_bonus_per_step = 2.0
        w.connectivity_bonus = 0.4

    # 2. ゲームモード (ソロ戦 vs 対人戦) に応じた調整
    if is_solo:
        # ソロプレイ時：相手からの妨害がないため、おじゃまや窒息に関するペナルティを 0 または極めて弱くする。
        w.buried_penalty = 0.0          # おじゃま評価を無効化（C++側で if(!oj.empty()) により自動スキップ）
        w.death_col_penalty = -0.1       # 窒息列ペナルティを最小限に（大連鎖構築の邪魔をしない）
    else:
        # 対人戦時：おじゃまぷよによる窒息死を徹底回避する。
        w.buried_penalty = -2.0          # おじゃまの下の埋もれを厳しく拒絶
        w.death_col_penalty = -2.5       # 窒息死を何よりも恐れる

    # 3. 停滞検出 (Stagnation) 時の緊急発火リセット重み
    if is_stagnated:
        w.chain_bonus_per_step = 15.0      # 即時発火を最大優先して盤面をリセット
        w.potential_score_scale = 0.1      # 将来的な伸ばしを一時諦める
    else:
        w.potential_score_scale = 1.0      # 通常時の潜在連鎖評価


class BeamSearchAgent(BasePlayerAgent):
    """
    Pure beam search agent — no neural network required.

    Expands all 22 placements for each of the next `look_ahead` tsumo pieces,
    retains the top `beam_width` boards at each depth, and returns the action
    leading to the highest-evaluated leaf.

    Parameters
    ----------
    beam_width : int
        Number of top candidate boards kept at every depth level.
        Higher = stronger but slower. 300-500 is typically fast enough.
    look_ahead : int
        How many tsumo pieces ahead to simulate (max 3 with standard preview).
    """

    def __init__(self, beam_width: int = 500, look_ahead: int = 3) -> None:
        self._cfg = p.BeamConfig()
        self._cfg.beam_width = beam_width
        self._cfg.look_ahead = look_ahead
        self._score_history = []
        self._is_solo = False
        _adjust_eval_weights(self._cfg, is_solo=False, is_stagnated=False)

    def adjust_for_mode(self, is_solo: bool) -> None:
        self._is_solo = is_solo

    def get_action(self, match, player_id: int, pres) -> p.Action:
        player = match.match.getPlayer(player_id)
        tsumo  = match.match.getTsumo()

        # 盤面の色ぷよ総数を集計 (おじゃま除く)
        puyo_count = 0
        for c in [p.Cell.Red, p.Cell.Green, p.Cell.Blue, p.Cell.Yellow]:
            puyo_count += player.field.getBitboard(c).popcount()

        # 停滞度の判定 (過去3手で期待スコアの伸びがなく、かつぷよ密度が4段以上の場合)
        is_stagnated = False
        if len(self._score_history) >= 3 and puyo_count >= 24:
            growth = self._score_history[-1] - self._score_history[-3]
            if growth <= 0.5:
                is_stagnated = True

        # 重みを動的調整
        _adjust_eval_weights(self._cfg, self._is_solo, is_stagnated)

        # 探索実行 (タプル (action_idx, expected_score) が返る)
        idx, expected_score = p.beam_search_action(player, tsumo, self._cfg)

        # スコア履歴を更新 (最大10手分保持)
        self._score_history.append(expected_score)
        if len(self._score_history) > 10:
            self._score_history.pop(0)

        return p.get_rl_action(idx)


# ---------------------------------------------------------------------------
# Hybrid: Beam Search + ONNX (Phase 2 preparation)
# ---------------------------------------------------------------------------
class HybridBeamOnnxAgent(BaseAIAgent):
    """
    Hybrid agent that combines beam search (chain construction) with ONNX
    policy inference (tactical / defensive decisions).

    Strategy:
      - When no ojama is incoming → use beam search to build chains.
      - When enemy ojama is pending → switch to ONNX for tactical response.

    This serves as the bridge between Phase 1 (beam search) and Phase 2 (MCTS).

    Parameters
    ----------
    model_path   : str   Path to the ONNX model file.
    beam_width   : int   Beam width for the construction phase.
    look_ahead   : int   Look-ahead depth for beam search.
    """

    def __init__(
        self,
        model_path: str,
        beam_width: int = 400,
        look_ahead: int = 3,
    ) -> None:
        super().__init__(model_path)
        self._beam_cfg = p.BeamConfig()
        self._beam_cfg.beam_width = beam_width
        self._beam_cfg.look_ahead = look_ahead
        self._score_history = []
        self._is_solo = False
        _adjust_eval_weights(self._beam_cfg, is_solo=False, is_stagnated=False)

    def adjust_for_mode(self, is_solo: bool) -> None:
        self._is_solo = is_solo

    def get_action(self, match, player_id: int, pres) -> p.Action:
        player = match.match.getPlayer(player_id)
        tsumo  = match.match.getTsumo()

        pending_ojama = int(player.active_ojama) + int(player.non_active_ojama)

        if pending_ojama == 0:
            # 盤面の色ぷよ総数を集計 (おじゃま除く)
            puyo_count = 0
            for c in [p.Cell.Red, p.Cell.Green, p.Cell.Blue, p.Cell.Yellow]:
                puyo_count += player.field.getBitboard(c).popcount()

            # 停滞度の判定
            is_stagnated = False
            if len(self._score_history) >= 3 and puyo_count >= 24:
                growth = self._score_history[-1] - self._score_history[-3]
                if growth <= 0.5:
                    is_stagnated = True

            # 重みを動的調整
            _adjust_eval_weights(self._beam_cfg, self._is_solo, is_stagnated)

            # Construction mode: build chains with beam search
            idx, expected_score = p.beam_search_action(player, tsumo, self._beam_cfg)

            # スコア履歴を更新
            self._score_history.append(expected_score)
            if len(self._score_history) > 10:
                self._score_history.pop(0)

            return p.get_rl_action(idx)
        else:
            # Tactical mode: use trained ONNX policy to respond to incoming ojama
            self._score_history.clear()
            obs_flat = p.build_observation(match.match)
            obs_flat = self._flip_observation_if_needed(obs_flat, player_id)
            return self._infer_action(obs_flat)
