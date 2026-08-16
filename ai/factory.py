"""
ai/factory.py

AgentFactory: Centralized Factory for instantiating AI and player agents from UI/CLI mode strings.
"""
from __future__ import annotations
from typing import Optional, Tuple

from .base import BasePlayerAgent
from .beam_agents import HumanPlayerAgent, EmptyPlayerAgent, BeamSearchAgent, VsBeamSearchAgent
from .negamax_agent import NegamaxAgent


class AgentFactory:
    """
    Translates player mode strings into concrete Agent instances.
    Provides standard player choices: Human, AI (VS Beam Search), and Empty (Solo).
    """

    MODE_HUMAN = "Human"
    MODE_AI = "AI"
    MODE_EMPTY_SOLO = "Empty (Solo)"

    # Clean, streamlined UI modes
    MODES: list[str] = [
        MODE_HUMAN,
        MODE_AI,
        MODE_EMPTY_SOLO,
    ]

    @classmethod
    def get_modes(cls, allow_empty: bool = True) -> list[str]:
        """Return available player modes, optionally filtering out Empty (Solo)."""
        if allow_empty:
            return list(cls.MODES)
        return [m for m in cls.MODES if m != cls.MODE_EMPTY_SOLO]

    @classmethod
    def create_agent(
        cls,
        mode: str,
        width: int | None = None,
        depth: int | None = None,
        dbs: int | None = None,
    ) -> Tuple[Optional[BasePlayerAgent], Optional[str]]:
        """
        Instantiate an agent configured with the given mode.
        Returns (agent, error_message).
        """
        # Primary standard modes
        if mode == cls.MODE_HUMAN:
            return HumanPlayerAgent(), None
        elif mode == cls.MODE_AI:
            return (
                VsBeamSearchAgent(
                    enable_attack_search=True,
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        elif mode == cls.MODE_EMPTY_SOLO:
            return EmptyPlayerAgent(), None

        # Backward-compatible aliases
        elif mode in ("AI: VS Beam (Gaze / Defense)", "AI: VS Beam (No Gaze)"):
            enable_attack = (mode == "AI: VS Beam (Gaze / Defense)")
            return (
                VsBeamSearchAgent(
                    enable_attack_search=enable_attack,
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        elif mode == "AI: BeamSearch (Solo / Normal)":
            return (
                BeamSearchAgent(
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        elif mode == "AI: Negamax (Alpha-Beta)":
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
