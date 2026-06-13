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
_CONFIG_PATH = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")


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
class BeamSearchAgent(BasePlayerAgent):
    """
    Pure beam search agent — no neural network required.

    Expands all placements for each of the next `look_ahead` tsumo pieces,
    retains the top `beam_width` boards at each depth, and returns the action
    leading to the highest-evaluated leaf.

    All JSON parsing, static caching, and profile overrides (such as solo_mode,
    vs_mode, deep_search, and stagnated) are managed entirely inside C++.
    """

    def __init__(self,
                 beam_width: int | None = None,
                 look_ahead: int | None = None) -> None:
        self._beam_width = beam_width
        self._look_ahead = look_ahead
        self._score_history = []
        self._is_solo = False

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

        width = self._beam_width if self._beam_width is not None else -1
        depth = self._look_ahead if self._look_ahead is not None else -1

        # C++側で設定ファイルをキャッシュ/ロードし、かつプロファイル(solo/vs/stagnated)を動的に適用
        idx, expected_score = p.beam_search_action(
            player, tsumo, _CONFIG_PATH, width, depth, self._is_solo, is_stagnated
        )

        # スコア履歴を更新 (最大10手分保持)
        self._score_history.append(expected_score)
        if len(self._score_history) > 10:
            self._score_history.pop(0)

        return p.get_rl_action(idx)



