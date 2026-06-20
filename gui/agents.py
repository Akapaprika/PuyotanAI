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
import threading

# Beam Search AI
# ---------------------------------------------------------------------------
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
        
        # Thread management for async non-blocking search
        self._thread: threading.Thread | None = None
        self._result: tuple[int, float] | None = None
        self._lock = threading.Lock()

    def adjust_for_mode(self, is_solo: bool) -> None:
        self._is_solo = is_solo

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        with self._lock:
            # If the background search finished and has a result, retrieve it and return
            if self._result is not None:
                idx, expected_score = self._result
                self._result = None
                self._thread = None
                
                # Update score history (keep up to 10 moves)
                self._score_history.append(expected_score)
                if len(self._score_history) > 10:
                    self._score_history.pop(0)
                
                return p.get_rl_action(idx)
            
            # If the search is still running, return None to wait (non-blocking)
            if self._thread is not None and self._thread.is_alive():
                return None

        player = match.match.getPlayer(player_id)
        tsumo  = match.match.getTsumo()

        # Count total puyos (including Ojama) on the board
        total_puyos = player.field.getOccupied().popcount()

        # Stagnation check (stagnated only when board is highly populated: >= 10 rows equivalent)
        is_stagnated = False
        if len(self._score_history) >= 4 and total_puyos >= 66:
            growth = self._score_history[-1] - self._score_history[-4]
            if growth <= 0.5:
                is_stagnated = True

        width = self._beam_width if self._beam_width is not None else -1
        depth = self._look_ahead if self._look_ahead is not None else -1

        # Define thread worker (GIL is released inside C++ bindings.cpp)
        def worker():
            res = p.beam_search_action(
                player, tsumo, _CONFIG_PATH, width, depth, self._is_solo, is_stagnated
            )
            with self._lock:
                self._result = res

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

        return None



