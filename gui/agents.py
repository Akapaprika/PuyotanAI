"""
gui/agents.py

Player Agent (Strategy) pattern.
Each agent is responsible for providing a p.Action when the engine marks
the player as needing a decision (decision_mask bit is set).

  HumanPlayerAgent  — waits for keyboard/button input buffered in pres state
  EmptyPlayerAgent  — immediately PASSes, creating an uncontested 1P side
  BeamSearchAgent   — performs heuristic-guided beam search simulation
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import puyotan_native as p

# ---------------------------------------------------------------------------
# beam_config.json のパス（C++ 側に渡すためだけに保持）
# ---------------------------------------------------------------------------
_CONFIG_PATH = str(Path(__file__).parent / "beam_config.json")


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
    """C++ BeamConfigLoader 経由で JSON を読み込み、プロファイルを差分適用する。"""
    # ベース値を JSON から C++ でロード
    loaded = p.load_beam_config(_CONFIG_PATH)
    cfg.eval_weights = loaded.eval_weights

    # apply_beam_profile は BeamConfig を値渡しで返すため、
    # 返り値の eval_weights を元の cfg に書き戻す。
    def _apply(profile_name: str) -> None:
        patched = p.apply_beam_profile(cfg, _CONFIG_PATH, profile_name)
        cfg.eval_weights = patched.eval_weights

    # 1. 探索の深さに応じたプロファイル適用
    if cfg.look_ahead >= 4:
        _apply("deep_search")

    # 2. ゲームモードに応じたプロファイル適用
    if is_solo:
        _apply("solo_mode")
    else:
        _apply("vs_mode")

    # 3. 停滞検出時のプロファイル適用（最後に上書き）
    if is_stagnated:
        _apply("stagnated")



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

    def __init__(self,
                 beam_width: int | None = None,
                 look_ahead: int | None = None) -> None:
        loaded = p.load_beam_config(_CONFIG_PATH)
        self._cfg = p.BeamConfig()
        self._cfg.beam_width = beam_width if beam_width is not None else loaded.beam_width
        self._cfg.look_ahead = look_ahead if look_ahead is not None else loaded.look_ahead
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



