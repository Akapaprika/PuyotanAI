"""
ai/beam_agents.py

Beam Search AI Player Agents (Solo / VS) and basic human/empty player adapters.
"""
from __future__ import annotations
from typing import Any

import puyotan_native as p
from .base import BasePlayerAgent, _AsyncSearchMixin, _STILL_THINKING
from .config import CONFIG_PATH


class HumanPlayerAgent(BasePlayerAgent):
    """
    Returns a PUT action when the user has confirmed their placement in the GUI
    (pres.confirmed == True), otherwise returns None to keep waiting.
    """

    is_human: bool = True

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        if pres is not None and getattr(pres, "confirmed", False):
            return p.Action(p.ActionType.PUT, pres.x, pres.rotation)
        return None


class EmptyPlayerAgent(BasePlayerAgent):
    """
    Mirrors the other player's PUT action exactly (solo / tokoton mode).
    """

    is_empty: bool = True

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        other_id = 1 - player_id
        other_state = match.get_player_state(other_id)
        other_action = other_state.current_action.action
        if other_action.type == p.ActionType.PUT:
            return p.Action(p.ActionType.PUT, other_action.x, other_action.rotation)
        return None


class BeamSearchAgent(_AsyncSearchMixin, BasePlayerAgent):
    """
    Solo / Unified Beam Search Agent.
    Executes solo_beam_search or vs_beam_search asynchronously in a background thread.
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
        self._last_result: tuple[int, float] | None = None
        self._init_async()

    def on_mode_updated(self, is_solo: bool) -> None:
        self._is_solo = is_solo

    @property
    def last_score(self) -> float:
        return self._last_result[1] if self._last_result is not None else 0.0

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        r = self._check_result()
        if r is _STILL_THINKING:
            return None
        if r is not None:
            self._last_result = r
            return p.get_rl_action(r[0])

        player = match.get_player_state(player_id)
        tsumo  = match.get_tsumo()

        width = self._beam_width      if self._beam_width      is not None else -1
        depth = self._look_ahead      if self._look_ahead      is not None else -1
        dbs   = self._dbs_max_similar if self._dbs_max_similar is not None else -1

        # Deep snapshots created on main thread
        player_snap = player.clone()
        tsumo_snap  = tsumo.clone()

        if self._is_solo:
            def worker():
                cfg = p.load_solo_config(CONFIG_PATH)
                if width > 0:  cfg.beam_width      = width
                if depth > 0:  cfg.look_ahead      = depth
                if dbs  >= 0:  cfg.dbs_max_similar = dbs
                res = p.solo_beam_search(player_snap, tsumo_snap, cfg, self._session)
                self._store_result(res)
        else:
            def worker():
                cfg = p.load_vs_config(CONFIG_PATH)
                if width > 0:  cfg.beam_width      = width
                if depth > 0:  cfg.look_ahead      = depth
                if dbs  >= 0:  cfg.dbs_max_similar = dbs
                res = p.vs_beam_search(player_snap, tsumo_snap, cfg, self._session)
                self._store_result(res)

        self._launch(worker)
        return None


class VsBeamSearchAgent(_AsyncSearchMixin, BasePlayerAgent):
    """
    VS Adversarial Beam Search Agent.
    Automatically populates VsEvalContext from the live game state each turn,
    enabling intelligent counter-attacks and threat avoidance.
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
        self._last_result: tuple[int, float] | None = None
        self._init_async()

    @property
    def last_score(self) -> float:
        return self._last_result[1] if self._last_result is not None else 0.0

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        r = self._check_result()
        if r is _STILL_THINKING:
            return None
        if r is not None:
            self._last_result = r
            return p.get_rl_action(r[0])

        player   = match.get_player_state(player_id)
        tsumo    = match.get_tsumo()
        enemy_id = 1 - player_id
        enemy    = match.get_player_state(enemy_id)

        bw  = self._beam_width      if self._beam_width      is not None else -1
        la  = self._look_ahead      if self._look_ahead      is not None else -1
        dbs = self._dbs_max_similar if self._dbs_max_similar is not None else -1

        cfg = p.load_vs_config(CONFIG_PATH)
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

        # Create deep snapshots on the main thread
        player_snap = player.clone()
        tsumo_snap  = tsumo.clone()

        def worker():
            res = p.vs_beam_search(player_snap, tsumo_snap, cfg, self._session)
            self._store_result(res)

        self._launch(worker)
        return None
