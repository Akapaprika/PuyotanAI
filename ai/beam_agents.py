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
    """Returns PUT action when confirmed in GUI, otherwise None."""
    is_human: bool = True

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        if pres is not None and getattr(pres, "confirmed", False):
            return p.Action(p.ActionType.PUT, pres.x, pres.rotation)
        return None


class EmptyPlayerAgent(BasePlayerAgent):
    """Mirrors the other player's PUT action (solo / tokoton mode)."""
    is_empty: bool = True

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        other_state = match.get_player_state(1 - player_id)
        act = other_state.current_action.action
        if act.type == p.ActionType.PUT:
            return p.Action(p.ActionType.PUT, act.x, act.rotation)
        return None


class _BaseBeamAgent(_AsyncSearchMixin, BasePlayerAgent):
    """Abstract base for beam search AI agents."""

    def __init__(self,
                 beam_width: int | None = None,
                 look_ahead: int | None = None,
                 dbs_max_similar: int | None = None) -> None:
        self._beam_width = beam_width
        self._look_ahead = look_ahead
        self._dbs_max_similar = dbs_max_similar
        self._session = p.BeamSearchSession()
        self._last_result: tuple[int, float] | None = None
        self._cfg: Any = None
        self._init_async()

    def _apply_overrides(self, cfg: Any) -> None:
        if self._beam_width and self._beam_width > 0:
            cfg.beam_width = self._beam_width
        if self._look_ahead and self._look_ahead > 0:
            cfg.look_ahead = self._look_ahead
        if self._dbs_max_similar is not None and self._dbs_max_similar >= 0:
            cfg.dbs_max_similar = self._dbs_max_similar
        if hasattr(cfg, 'recompute_beam_widths'):
            cfg.recompute_beam_widths()

    def reset(self) -> None:
        self.reset_search()
        self._session.reset()
        self._last_result = None
        self.reload_config()

    def reload_config(self) -> None:
        raise NotImplementedError

    def _start_search(self, match: Any, player_id: int) -> None:
        raise NotImplementedError

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        r = self._check_result()
        if r is _STILL_THINKING:
            return None
        if r is not None:
            self._last_result = r
            return p.get_rl_action(r[0])

        self._start_search(match, player_id)
        return None

    @property
    def last_score(self) -> float:
        return self._last_result[1] if self._last_result is not None else 0.0


class SoloBeamAgent(_BaseBeamAgent):
    """Dedicated Solo Beam Search Agent (Endless / Tokoton mode)."""

    def __init__(self,
                 beam_width: int | None = None,
                 look_ahead: int | None = None,
                 dbs_max_similar: int | None = None) -> None:
        super().__init__(beam_width, look_ahead, dbs_max_similar)
        self.reload_config()

    def reload_config(self) -> None:
        cfg = p.load_solo_config(CONFIG_PATH)
        self._apply_overrides(cfg)
        self._cfg = cfg

    def _start_search(self, match: Any, player_id: int) -> None:
        player_snap = match.get_player_state(player_id).clone()
        tsumo_snap  = match.get_tsumo().clone()
        cfg         = self._cfg

        def worker():
            res = p.solo_beam_search(player_snap, tsumo_snap, cfg, self._session)
            self._store_result(res)

        self._launch(worker)


class VsBeamAgent(_BaseBeamAgent):
    """Dedicated VS Adversarial Beam Search Agent (1v1 Match mode)."""

    def __init__(self,
                 enable_attack_search: bool = True,
                 beam_width: int | None = None,
                 look_ahead: int | None = None,
                 dbs_max_similar: int | None = None) -> None:
        self._enable_attack_search = enable_attack_search
        super().__init__(beam_width, look_ahead, dbs_max_similar)
        self.reload_config()

    def reload_config(self) -> None:
        cfg = p.load_vs_config(CONFIG_PATH)
        cfg.enable_attack_search = self._enable_attack_search
        self._apply_overrides(cfg)
        self._cfg = cfg

    def _start_search(self, match: Any, player_id: int) -> None:
        player = match.get_player_state(player_id)
        enemy  = match.get_player_state(1 - player_id)

        cfg = self._cfg
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

        player_snap = player.clone()
        tsumo_snap  = match.get_tsumo().clone()

        def worker():
            res = p.vs_beam_search(player_snap, tsumo_snap, cfg, self._session)
            self._store_result(res)

        self._launch(worker)


# Backward-compatibility aliases
BeamSearchAgent = SoloBeamAgent
VsBeamSearchAgent = VsBeamAgent
