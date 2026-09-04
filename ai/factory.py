from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple

from .base import BasePlayerAgent
from .beam_agents import (
    HumanPlayerAgent,
    EmptyPlayerAgent,
    SoloBeamAgent,
    VsBeamAgent,
    BeamSearchAgent,
    VsBeamSearchAgent,
)


class PlayerMode(str, Enum):
    """Type-safe enumeration for player agent modes."""
    HUMAN = "Human"
    AI    = "AI"
    EMPTY = "Empty (Solo)"

    def __str__(self) -> str:
        return self.value


class AgentFactory:
    """Translates player mode selections into concrete Agent instances."""
    MODE_HUMAN = PlayerMode.HUMAN
    MODE_AI = PlayerMode.AI
    MODE_EMPTY_SOLO = PlayerMode.EMPTY

    MODES: list[PlayerMode] = [
        PlayerMode.HUMAN,
        PlayerMode.AI,
        PlayerMode.EMPTY,
    ]

    @classmethod
    def get_modes(cls, allow_empty: bool = True) -> list[PlayerMode]:
        return list(cls.MODES) if allow_empty else [m for m in cls.MODES if m != PlayerMode.EMPTY]

    @classmethod
    def create_agent(
        cls,
        mode: PlayerMode | str,
        is_solo: bool = False,
        width: int | None = None,
        depth: int | None = None,
        dbs: int | None = None,
    ) -> Tuple[Optional[BasePlayerAgent], Optional[str]]:
        """Instantiate an agent configured with the given mode."""
        if isinstance(mode, str):
            # Backward-compatible legacy strings
            if mode in ("AI: VS Beam (Gaze / Defense)", "AI: VS Beam (No Gaze)"):
                return VsBeamAgent(enable_attack_search=(mode == "AI: VS Beam (Gaze / Defense)"),
                                   beam_width=width, look_ahead=depth, dbs_max_similar=dbs), None
            if mode == "AI: BeamSearch (Solo / Normal)":
                return SoloBeamAgent(beam_width=width, look_ahead=depth, dbs_max_similar=dbs), None
            try:
                mode = PlayerMode(mode)
            except ValueError:
                return None, f"Unknown mode: {mode}"

        if mode == PlayerMode.HUMAN:
            return HumanPlayerAgent(), None
        elif mode == PlayerMode.EMPTY:
            return EmptyPlayerAgent(), None
        elif mode == PlayerMode.AI:
            agent_cls = SoloBeamAgent if is_solo else VsBeamAgent
            return agent_cls(beam_width=width, look_ahead=depth, dbs_max_similar=dbs), None

        return None, f"Unknown mode: {mode}"

