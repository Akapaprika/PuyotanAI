"""
gui/agents.py

Player Agent (Strategy) pattern.
Each agent is responsible for providing a p.Action when the engine marks
the player as needing a decision (decision_mask bit is set).

  HumanPlayerAgent     — waits for keyboard/button input buffered in pres state
  EmptyPlayerAgent     — immediately PASSes, creating an uncontested 1P side
  BeamSearchAgent      — heuristic beam search (calls beam_search_action)
  VsBeamSearchAgent    — VS-mode beam search with explicit enable_attack_search
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import threading

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

    All JSON parsing, static caching, multi-turn stagnation tracking, and profile
    overrides are managed inside C++ using BeamSearchSession.
    """

    def __init__(self,
                 beam_width: int | None = None,
                 look_ahead: int | None = None,
                 dbs_max_similar: int | None = None,
                 is_enemy: bool | None = None) -> None:
        self._beam_width = beam_width
        self._look_ahead = look_ahead
        self._dbs_max_similar = dbs_max_similar
        self._is_enemy_override = is_enemy
        self._session = p.BeamSearchSession()
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
                idx, _ = self._result
                self._result = None
                self._thread = None
                return p.get_rl_action(idx)
            
            # If the search is still running, return None to wait (non-blocking)
            if self._thread is not None and self._thread.is_alive():
                return None

        player = match.match.getPlayer(player_id)
        tsumo  = match.match.getTsumo()

        width = self._beam_width if self._beam_width is not None else -1
        depth = self._look_ahead if self._look_ahead is not None else -1
        dbs = self._dbs_max_similar if self._dbs_max_similar is not None else -1

        if self._is_enemy_override is not None:
            is_enemy = self._is_enemy_override
        else:
            is_enemy = (not self._is_solo) and (player_id == 1)

        # Create deep snapshots of player and tsumo on the main thread (while GIL is held)
        player_snap = player.clone()
        tsumo_snap  = tsumo.clone()

        # Define thread worker (GIL is released inside C++ bindings.cpp)
        def worker():
            res = p.beam_search_action(
                player_snap, tsumo_snap, _CONFIG_PATH, width, depth, self._is_solo,
                dbs_max_similar=dbs, is_enemy=is_enemy, session=self._session
            )
            with self._lock:
                self._result = res

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

        return None


# ---------------------------------------------------------------------------
# VS Beam Search AI (explicit enable_attack_search flag)
# ---------------------------------------------------------------------------
class VsBeamSearchAgent(BasePlayerAgent):
    """
    VS-mode beam search agent that explicitly controls the enable_attack_search flag.

    Calls load_vs_config + vs_beam_search with BeamSearchSession for session state tracking.
    """

    def __init__(self,
                 enable_attack_search: bool = True,
                 beam_width: int | None = None,
                 look_ahead: int | None = None,
                 dbs_max_similar: int | None = None) -> None:
        self._enable_attack_search = enable_attack_search
        self._beam_width = beam_width
        self._look_ahead = look_ahead
        self._dbs_max_similar = dbs_max_similar
        self._session = p.BeamSearchSession()

        self._thread: threading.Thread | None = None
        self._result: tuple[int, float] | None = None
        self._lock = threading.Lock()

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        with self._lock:
            if self._result is not None:
                idx, _ = self._result
                self._result = None
                self._thread = None
                return p.get_rl_action(idx)

            if self._thread is not None and self._thread.is_alive():
                return None

        player     = match.match.getPlayer(player_id)
        tsumo      = match.match.getTsumo()
        enemy_id   = 1 - player_id
        enemy      = match.match.getPlayer(enemy_id)

        enable_attack = self._enable_attack_search
        bw  = self._beam_width    if self._beam_width    is not None else -1
        la  = self._look_ahead    if self._look_ahead    is not None else -1
        dbs = self._dbs_max_similar if self._dbs_max_similar is not None else -1

        cfg = p.load_vs_config(_CONFIG_PATH)
        cfg.enable_attack_search = enable_attack
        if bw  > 0: cfg.beam_width      = bw
        if la  > 0: cfg.look_ahead      = la
        if dbs >= 0: cfg.dbs_max_similar = dbs

        ctx = cfg.context
        ctx.enemy_field            = enemy.field
        ctx.enemy_active_next_pos  = enemy.active_next_pos
        ctx.enemy_action_type      = enemy.current_action.action.type
        ctx.enemy_chain_count      = enemy.chain_count
        ctx.enemy_score            = enemy.score
        ctx.enemy_used_score       = enemy.used_score
        ctx.enemy_active_ojama     = enemy.active_ojama
        ctx.enemy_non_active_ojama = enemy.non_active_ojama
        ctx.my_active_ojama        = player.active_ojama
        ctx.my_non_active_ojama    = player.non_active_ojama
        cfg.context = ctx

        # Create deep snapshots of player and tsumo on the main thread (while GIL is held)
        player_snap = player.clone()
        tsumo_snap  = tsumo.clone()

        def worker():
            res = p.vs_beam_search(player_snap, tsumo_snap, cfg, self._session)
            with self._lock:
                self._result = res

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return None


class NegamaxAgent(BasePlayerAgent):
    """
    Adversarial AI using Negamax (minimax) lookahead over deterministic PuyotanMatch states.
    Prunes move candidates at each decision turn using VS beam search, and simulates
    deterministic chain resolution / ojama mechanics via match.stepUntilDecision().
    """

    def __init__(self,
                 depth: int = 4,
                 candidate_n: int = 5,
                 beam_width: int | None = None,
                 look_ahead: int | None = None) -> None:
        self._depth = depth
        self._candidate_n = candidate_n
        self._beam_width = beam_width
        self._look_ahead = look_ahead

        self._thread: threading.Thread | None = None
        self._result: int | None = None
        self._lock = threading.Lock()

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        with self._lock:
            if self._result is not None:
                idx = self._result
                self._result = None
                self._thread = None
                return p.get_rl_action(idx)

            if self._thread is not None and self._thread.is_alive():
                return None

        # Snapshot match on main thread
        match_snap = match.match.clone()

        cfg = p.NegamaxConfig()
        cfg.depth = self._depth
        cfg.candidate_n = self._candidate_n

        vs_cfg = p.load_vs_config(_CONFIG_PATH)
        if self._beam_width is not None and self._beam_width > 0:
            vs_cfg.beam_width = self._beam_width
        if self._look_ahead is not None and self._look_ahead > 0:
            vs_cfg.look_ahead = self._look_ahead
        cfg.vs_config = vs_cfg

        def worker():
            res = p.negamax_search(match_snap, player_id, cfg)
            with self._lock:
                self._result = res.best_action

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return None

