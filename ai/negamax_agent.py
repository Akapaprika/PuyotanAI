"""
ai/negamax_agent.py

Adversarial Negamax search agent.
"""
from __future__ import annotations
from typing import Any

import puyotan_native as p
from .base import BasePlayerAgent, _AsyncSearchMixin, _STILL_THINKING
from .config import CONFIG_PATH


class NegamaxAgent(_AsyncSearchMixin, BasePlayerAgent):
    """
    Adversarial AI using Negamax (minimax) lookahead over deterministic PuyotanMatch states.
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

    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        r = self._check_result()
        if r is _STILL_THINKING:
            return None
        if r is not None:
            return p.get_rl_action(r)

        # Snapshot match on main thread
        match_snap = match.get_match_snapshot()

        cfg = p.load_negamax_config(CONFIG_PATH)
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
