"""
ai/factory.py

AgentFactory: Centralized Factory for instantiating AI and player agents from UI/CLI mode strings.
"""
from __future__ import annotations
from typing import Optional, Tuple

import puyotan_native as p
from .base import BasePlayerAgent
from .beam_agents import HumanPlayerAgent, EmptyPlayerAgent, BeamSearchAgent, VsBeamSearchAgent
from .negamax_agent import NegamaxAgent
from .config import CONFIG_PATH


class AgentFactory:
    """
    Translates player mode strings into concrete Agent instances.
    Provides default config values loaded from the central JSON configuration.
    """

    MODE_HUMAN = "Human"
    MODE_EMPTY_SOLO = "Empty (Solo)"
    MODE_BEAM_SEARCH_PLAYER = "AI: BeamSearch (Solo / Normal)"
    MODE_NEW_AI_ATTACK_ON = "AI: VS Beam (Gaze / Defense)"
    MODE_OLD_AI_ATTACK_OFF = "AI: VS Beam (No Gaze)"
    MODE_NEGAMAX = "AI: Negamax (Alpha-Beta)"

    MODES: list[str] = [
        MODE_HUMAN,
        MODE_EMPTY_SOLO,
        MODE_BEAM_SEARCH_PLAYER,
        MODE_NEW_AI_ATTACK_ON,
        MODE_OLD_AI_ATTACK_OFF,
        MODE_NEGAMAX,
    ]

    @classmethod
    def get_modes(cls, allow_empty: bool = True) -> list[str]:
        """Return available player modes, optionally filtering out Empty (Solo)."""
        if allow_empty:
            return list(cls.MODES)
        return [m for m in cls.MODES if m != cls.MODE_EMPTY_SOLO]

    @staticmethod
    def get_default_config() -> dict[str, int]:
        """Load default parameters from beam_config.json."""
        try:
            cfg = p.load_solo_config(CONFIG_PATH)
            return {
                "width": cfg.beam_width,
                "depth": cfg.look_ahead,
                "dbs": cfg.dbs_max_similar,
            }
        except Exception:
            return {"width": 1000, "depth": 15, "dbs": 6}

    @classmethod
    def create_agent(
        cls,
        mode: str,
        width: int | None = None,
        depth: int | None = None,
        dbs: int | None = None,
    ) -> Tuple[Optional[BasePlayerAgent], Optional[str]]:
        """Instantiate an agent configured with the given parameters. Returns (agent, error_message)."""
        if mode == cls.MODE_HUMAN:
            return HumanPlayerAgent(), None
        elif mode == cls.MODE_EMPTY_SOLO:
            return EmptyPlayerAgent(), None
        elif mode == cls.MODE_BEAM_SEARCH_PLAYER:
            return (
                BeamSearchAgent(
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        elif mode == cls.MODE_NEW_AI_ATTACK_ON:
            return (
                VsBeamSearchAgent(
                    enable_attack_search=True,
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        elif mode == cls.MODE_OLD_AI_ATTACK_OFF:
            return (
                VsBeamSearchAgent(
                    enable_attack_search=False,
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        elif mode == cls.MODE_NEGAMAX:
            return (
                NegamaxAgent(
                    depth=4,
                    candidate_n=5,
                    beam_width=width,
                    look_ahead=depth,
                ),
                None,
            )
        else:
            return None, f"Unknown mode: {mode}"
