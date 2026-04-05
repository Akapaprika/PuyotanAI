"""
gui/agents.py

Player Agent (Strategy) pattern.
Each agent is responsible for providing a p.Action when the engine marks
the player as needing a decision (decision_mask bit is set).

  HumanPlayerAgent  — waits for keyboard/button input buffered in pres state
  AIPlayerAgent     — runs ONNX inference via puyotan_native.OnnxPolicy
  EmptyPlayerAgent  — immediately PASSes, creating an uncontested 1P side
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import puyotan_native as p


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
# AI (ONNX)
# ---------------------------------------------------------------------------
class AIPlayerAgent(BasePlayerAgent):
    """
    Runs ONNX inference on every decision point.
    The action is chosen immediately — no human latency.

    Parameters
    ----------
    model_path : str  Path to the .onnx model file.
    """

    # Observation size constants (must match ObservationBuilder in C++)
    OBS_SIZE = 2 * 5 * 6 * 14  # kBytesPerObservation

    # All 22 PUT actions, in exactly the same order as the C++ training loop.
    # Built from p.get_rl_action() so this is always in sync with the engine.
    #
    # Layout (w=6):
    #   [ 0.. 5]  Up,    col 0-5
    #   [ 6..10]  Right, col 0-4
    #   [11..16]  Down,  col 0-5
    #   [17..21]  Left,  col 0-4
    _VALID_ACTIONS: list[p.Action] = [
        p.get_rl_action(i) for i in range(p.kNumRLActions)
    ]

    def __init__(self, model_path: str) -> None:
        self.policy = p.OnnxPolicy(model_path, use_cpu=True)
        self._obs_buf = np.zeros(self.OBS_SIZE, dtype=np.uint8)

    def get_action(self, match, player_id: int, pres) -> p.Action:
        # Build observation tensor
        obs_flat = p.build_observation(match.match)  # shape (OBS_SIZE,)

        # Flip perspective for P2: swap the two player halves of the obs
        half = self.OBS_SIZE // 2
        if player_id == 1:
            obs_flat = np.concatenate([obs_flat[half:], obs_flat[:half]])

        # Run inference — infer() expects a batch, so add batch dim
        obs_batch = obs_flat.reshape(1, -1)
        action_indices = self.policy.infer(obs_batch)  # list of int
        idx = int(action_indices[0])
        idx = max(0, min(idx, len(self._VALID_ACTIONS) - 1))
        return self._VALID_ACTIONS[idx]

    def reload(self, model_path: str) -> None:
        """Hot-swap the underlying ONNX model."""
        self.policy = p.OnnxPolicy(model_path, use_cpu=True)


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
