"""
gui/agents.py

Player Agent (Strategy) pattern.
Each agent is responsible for providing a p.Action when the engine marks
the player as needing a decision (decision_mask bit is set).

  HumanPlayerAgent    — waits for keyboard/button input buffered in pres state
  EmptyPlayerAgent    — immediately PASSes, creating an uncontested 1P side
  BeamSearchAgent     — solo beam search (soloBeamSearch / vsBeamSearch via is_solo flag)
  VsBeamSearchAgent   — VS beam search with explicit enable_attack_search flag + context
  NegamaxAgent        — Negamax adversarial search over PuyotanMatch states
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import threading

import puyotan_native as p
from . import config

# Sentinel returned by _AsyncSearchMixin._check_result() while a thread is alive.
_STILL_THINKING = object()


# ---------------------------------------------------------------------------
# Async search thread management mixin
# ---------------------------------------------------------------------------
class _AsyncSearchMixin:
    """
    Reusable thread-management helpers for AI agents that run search in a background
    thread. Call _init_async() from __init__ before using any other method.
    """

    def _init_async(self) -> None:
        self._thread: threading.Thread | None = None
        self._result: Any = None
        self._lock = threading.Lock()

    def _check_result(self):
        """
        Non-blocking poll of the background search.

        Returns
        -------
        result          — the stored value if search finished (clears state).
        _STILL_THINKING — a search thread is alive but not yet done.
        None            — no thread is running; caller should start one.
        """
        with self._lock:
            if self._result is not None:
                r, self._result, self._thread = self._result, None, None
                return r
            if self._thread is not None and self._thread.is_alive():
                return _STILL_THINKING
        return None

    def _launch(self, target) -> None:
        """Start a new daemon search thread."""
        t = threading.Thread(target=target, daemon=True)
        with self._lock:
            self._thread = t
        t.start()

    def _store_result(self, result) -> None:
        """Called from inside the worker thread to publish the result."""
        with self._lock:
            self._result = result


# ---------------------------------------------------------------------------
# Base agent interface
# ---------------------------------------------------------------------------
class BasePlayerAgent(ABC):
    """Abstract interface for a player controller."""

    # Subclasses override these as class-level flags to avoid isinstance checks.
    is_human: bool = False
    is_empty: bool = False

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

    def on_mode_updated(self, is_solo: bool) -> None:
        """
        Called each frame before get_action() when the game mode context changes.
        Default is a no-op; override in subclasses that adapt behaviour to solo/vs mode.
        """


# ---------------------------------------------------------------------------
# Human
# ---------------------------------------------------------------------------
class HumanPlayerAgent(BasePlayerAgent):
    """
    Returns a PUT action when the user has confirmed their placement
    (pres.confirmed == True), otherwise returns None to keep waiting.
    """

    is_human: bool = True

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

    is_empty: bool = True

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
# Beam Search AI (Solo / VS共用)
# ---------------------------------------------------------------------------
class BeamSearchAgent(_AsyncSearchMixin, BasePlayerAgent):
    """
    Pure beam search agent — no neural network required.

    Expands all placements for each of the next `look_ahead` tsumo pieces,
    retains the top `beam_width` boards at each depth, and returns the action
    leading to the highest-evaluated leaf.

    Multi-turn session state (stagnation detection) is managed inside C++
    using BeamSearchSession.
    """

    def __init__(self,
                 beam_width: int | None = None,
                 look_ahead: int | None = None,
                 dbs_max_similar: int | None = None) -> None:
        self._beam_width = beam_width
        self._look_ahead = look_ahead
        self._dbs_max_similar = dbs_max_similar
        self._session = p.BeamSearchSession()
        self._is_solo = False
        self._init_async()

    def on_mode_updated(self, is_solo: bool) -> None:
        self._is_solo = is_solo

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        r = self._check_result()
        if r is _STILL_THINKING:
            return None
        if r is not None:
            return p.get_rl_action(r[0])

        player = match.get_player_state(player_id)
        tsumo  = match.get_tsumo()

        width = self._beam_width      if self._beam_width      is not None else -1
        depth = self._look_ahead      if self._look_ahead      is not None else -1
        dbs   = self._dbs_max_similar if self._dbs_max_similar is not None else -1

        # Create deep snapshots on the main thread (while GIL is held)
        player_snap = player.clone()
        tsumo_snap  = tsumo.clone()

        # Define thread worker (GIL is released inside C++ bindings.cpp)
        def worker():
            res = p.beam_search_action(
                player_snap, tsumo_snap, config.CONFIG_PATH, width, depth,
                self._is_solo,
                dbs_max_similar=dbs, session=self._session
            )
            self._store_result(res)

        self._launch(worker)
        return None


# ---------------------------------------------------------------------------
# VS Beam Search AI (explicit enable_attack_search flag + VsEvalContext)
# ---------------------------------------------------------------------------
class VsBeamSearchAgent(_AsyncSearchMixin, BasePlayerAgent):
    """
    VS-mode beam search agent that explicitly controls the enable_attack_search flag.

    Calls load_vs_config + vs_beam_search with BeamSearchSession for session state tracking.
    Automatically populates VsEvalContext from the live match state each turn.
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
        self._init_async()

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        r = self._check_result()
        if r is _STILL_THINKING:
            return None
        if r is not None:
            return p.get_rl_action(r[0])

        player   = match.get_player_state(player_id)
        tsumo    = match.get_tsumo()
        enemy_id = 1 - player_id
        enemy    = match.get_player_state(enemy_id)

        bw  = self._beam_width      if self._beam_width      is not None else -1
        la  = self._look_ahead      if self._look_ahead      is not None else -1
        dbs = self._dbs_max_similar if self._dbs_max_similar is not None else -1

        cfg = p.load_vs_config(config.CONFIG_PATH)
        cfg.enable_attack_search = self._enable_attack_search
        if bw  > 0:  cfg.beam_width      = bw
        if la  > 0:  cfg.look_ahead      = la
        if dbs >= 0: cfg.dbs_max_similar = dbs

        # Populate VsEvalContext from live match state
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

        # Create deep snapshots on the main thread (while GIL is held)
        player_snap = player.clone()
        tsumo_snap  = tsumo.clone()

        def worker():
            res = p.vs_beam_search(player_snap, tsumo_snap, cfg, self._session)
            self._store_result(res)

        self._launch(worker)
        return None


# ---------------------------------------------------------------------------
# Negamax AI (Matchを用いた先読み対戦探索)
# ---------------------------------------------------------------------------
class NegamaxAgent(_AsyncSearchMixin, BasePlayerAgent):
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
        self._init_async()

    def get_action(self, match, player_id: int, pres) -> p.Action | None:
        r = self._check_result()
        if r is _STILL_THINKING:
            return None
        if r is not None:
            return p.get_rl_action(r)

        # Snapshot match on main thread (safe to clone while GIL is held)
        match_snap = match.get_match_snapshot()

        cfg = p.load_negamax_config(config.CONFIG_PATH)
        if self._depth is not None and self._depth > 0:
            cfg.depth = self._depth
        if self._candidate_n is not None and self._candidate_n > 0:
            cfg.candidate_n = self._candidate_n
        if self._beam_width is not None and self._beam_width > 0:
            cfg.vs_config.beam_width = self._beam_width
        if self._look_ahead is not None and self._look_ahead > 0:
            cfg.vs_config.look_ahead = self._look_ahead

        def worker():
            res = p.negamax_search(match_snap, player_id, cfg)
            self._store_result(res.best_action)

        self._launch(worker)
        return None
