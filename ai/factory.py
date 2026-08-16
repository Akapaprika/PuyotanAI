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
    """
    Type-safe enumeration for player agent modes.
    Inherits from str so that it formats cleanly in UI and serializes naturally.
    """
    HUMAN = "Human"
    AI    = "AI"
    EMPTY = "Empty (Solo)"

    def __str__(self) -> str:
        return self.value


class AgentFactory:
    """
    Translates player mode selections into concrete Agent instances.
    """

    # Aliases for backward compatibility
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
        """Return available player modes, optionally filtering out Empty."""
        if allow_empty:
            return list(cls.MODES)
        return [m for m in cls.MODES if m != PlayerMode.EMPTY]

    @classmethod
    def create_agent(
        cls,
        mode: PlayerMode | str,
        is_solo: bool = False,
        width: int | None = None,
        depth: int | None = None,
        dbs: int | None = None,
    ) -> Tuple[Optional[BasePlayerAgent], Optional[str]]:
        """
        Instantiate an agent configured with the given mode.
        If is_solo is True (e.g. playing against Empty), PlayerMode.AI instantiates a SoloBeamAgent.
        Returns (agent, error_message).
        """
        # Convert string to PlayerMode if standard value
        if isinstance(mode, str):
            try:
                mode_enum = PlayerMode(mode)
            except ValueError:
                mode_enum = None
        else:
            mode_enum = mode

        # Primary standard modes
        if mode_enum == PlayerMode.HUMAN:
            return HumanPlayerAgent(), None
        elif mode_enum == PlayerMode.EMPTY:
            return EmptyPlayerAgent(), None
        elif mode_enum == PlayerMode.AI:
            if is_solo:
                return (
                    SoloBeamAgent(
                        beam_width=width,
                        look_ahead=depth,
                        dbs_max_similar=dbs,
                    ),
                    None,
                )
            else:
                return (
                    VsBeamAgent(
                        enable_attack_search=True,
                        beam_width=width,
                        look_ahead=depth,
                        dbs_max_similar=dbs,
                    ),
                    None,
                )

        # Backward-compatible string aliases
        if mode in ("AI: VS Beam (Gaze / Defense)", "AI: VS Beam (No Gaze)"):
            enable_attack = (mode == "AI: VS Beam (Gaze / Defense)")
            return (
                VsBeamAgent(
                    enable_attack_search=enable_attack,
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        elif mode == "AI: BeamSearch (Solo / Normal)":
            return (
                SoloBeamAgent(
                    beam_width=width,
                    look_ahead=depth,
                    dbs_max_similar=dbs,
                ),
                None,
            )
        else:
            return None, f"Unknown mode: {mode}"
