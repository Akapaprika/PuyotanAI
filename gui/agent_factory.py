"""
gui/agent_factory.py

Factory for player agents.
Decouples UI widgets from concrete agent implementations and default parameter loading.
"""
from __future__ import annotations

from typing import Any
import puyotan_native as p
from . import config
from .agents import (
    BasePlayerAgent,
    HumanPlayerAgent,
    EmptyPlayerAgent,
    BeamSearchAgent,
    VsBeamSearchAgent,
    NegamaxAgent,
)


class AgentFactory:
    """Creates player agents based on mode labels and user parameters."""

    MODE_HUMAN = "Human"
    MODE_NEGAMAX = "Negamax AI (Lookahead)"
    MODE_NEW_AI_ATTACK_ON = "New AI (Attack ON)"
    MODE_OLD_AI_ATTACK_OFF = "Old AI (Attack OFF)"
    MODE_BEAM_SEARCH_PLAYER = "Beam Search (Player)"
    MODE_EMPTY_SOLO = "Empty (Solo)"

    ALL_MODES = [
        MODE_HUMAN,
        MODE_NEGAMAX,
        MODE_NEW_AI_ATTACK_ON,
        MODE_OLD_AI_ATTACK_OFF,
        MODE_BEAM_SEARCH_PLAYER,
        MODE_EMPTY_SOLO,
    ]

    @classmethod
    def get_modes(cls, allow_empty: bool = True) -> list[str]:
        """Returns the list of selectable agent mode names."""
        return list(cls.ALL_MODES if allow_empty else cls.ALL_MODES[:-1])

    @classmethod
    def get_default_config(cls) -> dict[str, int]:
        """Load default beam parameters from the canonical beam_config.json."""
        defaults = {"width": 15000, "depth": 25, "dbs": 6}
        try:
            cfg = p.load_solo_config(config.CONFIG_PATH)
            defaults["width"] = cfg.beam_width
            defaults["depth"] = cfg.look_ahead
            defaults["dbs"] = cfg.dbs_max_similar
        except Exception:
            pass
        return defaults

    @classmethod
    def create_agent(
        cls,
        mode: str,
        width: int = 15000,
        depth: int = 25,
        dbs: int = 6,
    ) -> tuple[BasePlayerAgent | None, str | None]:
        """
        Instantiate and return (agent, None) on success, or (None, error_message) on failure.
        """
        if mode == cls.MODE_HUMAN:
            return HumanPlayerAgent(), None
        if mode == cls.MODE_NEGAMAX:
            return NegamaxAgent(depth=depth, candidate_n=22, beam_width=width, look_ahead=3), None
        if mode == cls.MODE_NEW_AI_ATTACK_ON:
            return VsBeamSearchAgent(enable_attack_search=True, beam_width=width, look_ahead=depth, dbs_max_similar=dbs), None
        if mode == cls.MODE_OLD_AI_ATTACK_OFF:
            return VsBeamSearchAgent(enable_attack_search=False, beam_width=width, look_ahead=depth, dbs_max_similar=dbs), None
        if mode == cls.MODE_BEAM_SEARCH_PLAYER:
            return BeamSearchAgent(beam_width=width, look_ahead=depth, dbs_max_similar=dbs), None
        if mode == cls.MODE_EMPTY_SOLO:
            return EmptyPlayerAgent(), None

        return None, f"Unknown mode: {mode}"
